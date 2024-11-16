<img src="https://yuheng.ink/project-page/control-3d-scene/images/logo.png" height="75px" align="left"> 

# Controllable 3D Outdoor Scene Generation

[![Visitors](https://api.visitorbadge.io/api/visitors?path=yuheng-control-3d-scene&label=Visitors&countColor=%23fedcba&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=yuheng-control-3d-scene)

![Teaser](https://yuheng.ink/project-page/control-3d-scene/images/teaser.jpg)

Three-dimensional scene generation is crucial in computer vision, with applications spanning autonomous driving, gaming and the metaverse. Current methods either lack user control or rely on imprecise, non-intuitive conditions. In this work, we propose a method that uses scene graphs—an accessible, user-friendly control format—to generate outdoor 3D scenes. We develop an interactive system that transforms a sparse scene graph into a dense BEV (Bird's Eye View) Embedding Layout, which guides a conditional diffusion model to generate 3D scenes that match the scene graph description. During inference, users can easily create or modify scene graphs to generate large-scale outdoor scenes. We create a large-scale dataset with paired scene graphs and 3D semantic scenes to train the BEV embedding and diffusion models. Experimental results show that our approach consistently produces high-quality 3D urban scenes closely aligned with the input scene graphs. 

## NEWS

- [2024/11/15] Official repo is created, code will be released soon.
