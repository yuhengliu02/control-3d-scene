import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-30

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    m = torch.max(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f"{x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    return torch.log(x_onehot.float().clamp(min=1e-30))

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    ac = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    alphas = ac[1:] / ac[:-1]
    alphas = np.sqrt(np.clip(alphas, 0.001, 1.0))
    return alphas

class ConditionalDiffusion(torch.nn.Module):
    def __init__(self, denoise_fn, num_classes=11, num_timesteps=100,
                 auxiliary_loss_weight=1e-4, adaptive_auxiliary_loss=True):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss

        alphas = torch.tensor(cosine_beta_schedule(num_timesteps).astype("float64"))
        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        self.register_buffer("log_alpha", log_alpha.float())
        self.register_buffer("log_1_min_alpha", log_1_min_alpha.float())
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha.float())
        self.register_buffer("log_1_min_cumprod_alpha", log_1_min_cumprod_alpha.float())
        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    def multinomial_kl(self, log_p1, log_p2):
        return (log_p1.exp() * (log_p1 - log_p2)).sum(dim=1)

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        return log_add_exp(log_x_t + log_alpha_t,
                           log_1_min_alpha_t - np.log(self.num_classes))

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        return log_add_exp(log_x_start + log_cumprod_alpha_t,
                           log_1_min_cumprod_alpha - np.log(self.num_classes))

    def predict_start(self, log_x_t, t, cond):
        x_t = log_onehot_to_index(log_x_t)
        out = self.denoise_fn(x_t, cond, t)
        return F.log_softmax(out, dim=1)

    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = torch.where(t - 1 < 0, torch.zeros_like(t), t - 1)
        log_EV = self.q_pred(log_x_start, t_minus_1)
        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV = torch.where(t_broadcast == 0, log_x_start, log_EV)
        unnormed = log_EV + self.q_pred_one_timestep(log_x_t, t)
        return unnormed - torch.logsumexp(unnormed, dim=1, keepdim=True)

    def p_pred(self, log_x, t, cond):
        log_x0_recon = self.predict_start(log_x, t, cond)
        log_model_pred = self.q_posterior(log_x0_recon, log_x, t)
        return log_model_pred, log_x0_recon

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + EPS) + EPS)
        sample = (gumbel_noise + logits).argmax(dim=1)
        return index_to_log_onehot(sample, self.num_classes)

    def q_sample(self, log_x_start, t):
        return self.log_sample_categorical(self.q_pred(log_x_start, t))

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        ones = torch.ones(b, device=log_x_start.device).long()
        log_qxT = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half = -torch.log(self.num_classes * torch.ones_like(log_qxT))
        return sum_except_batch(self.multinomial_kl(log_qxT, log_half))

    def sample_time(self, b, device, method="importance"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 1e-4
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            return t, pt_all.gather(dim=0, index=t)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return t, torch.ones_like(t).float() / self.num_timesteps

    def forward(self, x, cond):
        b, device = x.size(0), x.device
        spatial = x.size()[1:]
        t, pt = self.sample_time(b, device, "importance")

        log_x_start = index_to_log_onehot(x, self.num_classes)
        log_x_t = self.q_sample(log_x_start, t)

        log_true_prob = self.q_posterior(log_x_start, log_x_t, t)
        log_model_prob, log_x0_recon = self.p_pred(log_x_t, t, cond)

        kl = sum_except_batch(self.multinomial_kl(log_true_prob, log_model_prob))
        decoder_nll = sum_except_batch(-log_categorical(log_x_start, log_model_prob))
        mask = (t == 0).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl

        if self.training:
            Lt2 = kl_loss.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_hist = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_hist)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        loss = kl_loss / pt + self.kl_prior(log_x_start)

        kl_aux = sum_except_batch(
            self.multinomial_kl(log_x_start[:, :-1], log_x0_recon[:, :-1]))
        kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
        weight = (1 - t / self.num_timesteps) + 1.0 if self.adaptive_auxiliary_loss else 1.0
        loss = loss + weight * self.auxiliary_loss_weight * kl_aux_loss / pt

        return loss.sum() / (spatial[0] * spatial[1])

    @torch.no_grad()
    def sample(self, cond, num_steps=None):
        device = self.log_alpha.device
        b = cond.size(0)
        spatial = cond.size()[2:]
        num_steps = num_steps or self.num_timesteps

        log_z = self.log_sample_categorical(
            torch.zeros((b, self.num_classes) + tuple(spatial), device=device))
        for i in reversed(range(num_steps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_model_prob, _ = self.p_pred(log_z, t, cond)
            log_z = self.log_sample_categorical(log_model_prob)
        return log_onehot_to_index(log_z)
