<img src="https://yuheng.ink/project-page/control-3d-scene/images/logo.png" height="75px" align="left"> 

# Controllable 3D Outdoor Scene Generation via Scene Graphs

[ICCV 2025] [Yuheng Liu](https://yuheng.ink/)<sup>1,2</sup>, [Xinke Li](https://shinke-li.github.io/)<sup>3</sup>, [Yuning Zhang](https://scholar.google.com/citations?hl=en&user=nbvkScUAAAAJ)<sup>4</sup>, [Lu Qi](http://luqi.info/)<sup>5</sup>, [Xin Li](https://github.com/yuhengliu02/control-3d-scene)<sup>1</sup>, [Wenping Wang](https://github.com/yuhengliu02/control-3d-scene)<sup>1</sup>, [Chongshou Li](https://scholar.google.com.sg/citations?user=pQsr70EAAAAJ&hl=en)<sup>4</sup>, [Xueting Li](https://sunshineatnoon.github.io/)<sup>6*</sup>, [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en&oi=ao)<sup>2*</sup>

<sup>1</sup>Texas A&M University, <sup>2</sup>The University of Cailfornia, Merced, <sup>3</sup>City University of HongKong, <sup>4</sup>Southwest Jiaotong University, <sup>5</sup>Insta360 Research, <sup>6</sup>NVIDIA

[![Visitors](https://api.visitorbadge.io/api/visitors?path=yuheng-control-3d-scene&label=Visitors&countColor=%23fedcba&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=yuheng-control-3d-scene)  [![Static Badge](https://img.shields.io/badge/PDF-Download-red?logo=Adobe%20Acrobat%20Reader)](https://yuheng.ink/project-page/control-3d-scene/papers/controllable_3d_outdoor_scene_generation_via_scene_graphs.pdf)  [![Static Badge](https://img.shields.io/badge/2503.07152-b31b1b?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2503.07152)  [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?logo=Google%20Chrome&logoColor=white)](https://yuheng.ink/project-page/control-3d-scene/)  [![Static Badge](https://img.shields.io/badge/Youtube-%23ff0000?style=flat&logo=Youtube)](https://www.youtube.com/watch?v=zu1-FbK9ETc)  

![Teaser](https://yuheng.ink/project-page/control-3d-scene/images/teaser.jpg)

Three-dimensional scene generation is crucial in computer vision, with applications spanning autonomous driving, gaming and the metaverse. Current methods either lack user control or rely on imprecise, non-intuitive conditions. In this work, we propose a method that uses scene graphs—an accessible, user-friendly control format—to generate outdoor 3D scenes. We develop an interactive system that transforms a sparse scene graph into a dense BEV (Bird's Eye View) Embedding Layout, which guides a conditional diffusion model to generate 3D scenes that match the scene graph description. During inference, users can easily create or modify scene graphs to generate large-scale outdoor scenes. We create a large-scale dataset with paired scene graphs and 3D semantic scenes to train the BEV embedding and diffusion models. Experimental results show that our approach consistently produces high-quality 3D urban scenes closely aligned with the input scene graphs. 

## NEWS

- [2025/06/25] Our work is accepted by ICCV 2025.
- [2025/03/10] Our work is now on [arXiv](https://arxiv.org/abs/2503.07152).
- [2024/11/15] Official repo is created, code will be released soon.
