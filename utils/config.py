import argparse

import yaml

class Config(argparse.Namespace):

    def get(self, key, default=None):
        return getattr(self, key, default)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    base = argparse.ArgumentParser()
    base.add_argument("--config", "-c", type=str, required=True)
    base.add_argument("--exp_name", "-n", type=str, default="default")
    known, _ = base.parse_known_args()

    cfg = load_config(known.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--exp_name", "-n", type=str, default="default")
    for key, val in cfg.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda s: s.lower() == "true", default=val)
        elif isinstance(val, list):
            parser.add_argument(f"--{key}", type=type(val[0]) if val else str,
                                nargs="+", default=val)
        elif val is None:
            parser.add_argument(f"--{key}", default=None)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args(namespace=Config())
