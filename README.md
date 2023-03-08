# 3DBoxIdent: Identifiability Benchmark Datasets

[[_3DBoxIdent_ part 1]()][[_3DBoxIdent_ part 2]()][[Paper](https://openreview.net/forum?id=U_2kuqoTcB)]

Official code base for the generation of the _3DBoxIdent_ datasets presented in the paper [Identifiability Results for Multimodal Contrastive
Learning](https://openreview.net/forum?id=U_2kuqoTcB) and featured at the [CLeaR datasets track 2023](https://www.cclear.cc/2023/CallforDatasets). This GitHub repository extends the [3DIdent](https://arxiv.org/pdf/2102.08850.pdf) dataset and builds on top of its [generation code](https://github.com/brendel-group/cl-ica). The _3DBoxIdent_ datasets offer an identifiability benchmark by providing image/image pairs generated from controlled ground-truth factors using the [Blender](https://github.com/blender/blender) software. Ground-truth factors are either independantly sampled (_part 1_) or non-trivial dependencies exist between factors (_part 2_). 

<p align="center">
  <img src="https://github.com/alicebizeul/Multimodal3DIdent/blob/main/sample_image.png" alt="Multiviewl3DIdent dataset example images" width=570 />
</p>

This repository offers the possibility to adjust generative assumptions to fit personal preferences and generate custom-made datasets. 

## Dataset Description
- - -
The training/validation/test datasets consist of 250,000/10,000/10,000
samples of image pairs respectively. Images depict a colored teapot in front of a colored background, illuminated by a colored spotlight. _View 1_ displays the teaport with a **metallic** texture; _View 2_ displays the teapot with a **rubber** texture.

Each image is controlled by _9_ factors of variation partitioned into _3_ information blocks (_Rotations, Positions, Hues_) detailed below: 

| Information Block      | Description | Raw Support | Blender Support | Details |
| ---------- | ----------- |-------- |--------     |-------- |
| Rotation   | $\alpha$-angle        | $[-\pi;\pi]$| $[-1;1]$   | Object $\alpha$ rotation angle|
| Rotation   | $\beta$-angle         | $[-\pi;\pi]$| $[-1;1]$   | Object $\beta$ rotation angle|
| Rotation   | $\gamma$-angle        | $[-\pi;\pi]$| $[-1;1]$   | Spotlight rotation angle|
| Position   | $x$-coordinate        | $[-1;1]$| $[-2;2]$       | Object $x$-coordinate|
| Position   | $y$-coordinate        | $[-1;1]$| $[-2;2]$       | Object $y$-coordinate|
| Position   | $z$-coordinate        | $[-1;1]$| $[-2;2]$       | Object $z$-coordinate|
| Hue        | Object color          | $[-1;1]$| $[-\pi; \pi]$  | Object HSV color, H parameter|
| Hue        | Background color      | $[-1;1]$| $[-\pi; \pi]$  | Background HSV color, H parameter |
| Hue        | Spotlight color       | $[-1;1]$| $[-\pi; \pi]$  | Spotlight HSV color, H parameter|

The generation of ground truth factors pairs follows specific distributions according to three sets of rules (i.e., <span style="color: RoyalBlue;">Block Type</span>). Factors following the _content_  <span style="color: RoyalBlue;">Block Type</span> are _shared_ accross views. Factors following the _style_ <span style="color: RoyalBlue;">Block Type</span> are stochastic between views. Factors following the view-specific <span style="color: RoyalBlue;">Block Type</span> are either kept constant or specific to one view. For each dataset, <span style="color: Crimson;">Information Blocks</span> are associated with a unique <span style="color: RoyalBlue;">Block Type</span>. Distributions linked to each <span style="color: RoyalBlue;">Block Type</span> are detailed below for _3DBoxIdent_. <span style="color: RoyalBlue;">Block Type</span>

All combinaisons of _<span style="color: Crimson;">Information Blocks</span> : <span style="color: RoyalBlue;">Block Type</span>_ are available for download.

### Part 1: Without inter- & intra- block causal dependencies

| Block Type | Symbol          | View 1 Distribution | View 2 Distribution | Description         |
|----------- |------------------- |-------------------- |-------------------- |-------------------- |
| Content    |  $c=[c_1,c_2,c_3]$ |  $c \sim [\mathcal{U}([-1,1]),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{c} \sim [\mathcal{\delta}(\tilde{c_1}-c_1),\mathcal{\delta}(\tilde{c_2}-c_2),\mathcal{\delta}(\tilde{c_3}-c_3)]$  | Shared between views |
| Style      |  $s=[s_1,s_2,s_3]$ |  $s \sim [\mathcal{U}([-1,1]),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{s} \sim {\mathcal{N}_{[-1,1]}(s_1,1),\mathcal{N}_{[-1,1]}(s_2,1),\mathcal{N}_{[-1,1]}(s_3,1)}$ | Stochastic between views |
| View-Specific|$m=[m_1,m_2,m_3]$ |  $m \sim [\mathcal{U}([-1,1]),\mathcal{\delta}(0),\mathcal{\delta}(0)]$  |   $\tilde{m} \sim [\mathcal{\delta}(0),\mathcal{\delta}(0),\mathcal{U}([-1,1])]$  | $m_1$ is specific to _View 1_, $m_2$ is constant, $m_3$ is specific to _View 2_ |

### Part 2: With inter- & intra- block causal dependencies

| Block Type |    Symbol   | View 1 Distribution               | View 2 Distribution                   | Description          | 
|----------  |------------ |---------------------------------- |-------------------------------------- |--------------------- |
| Content    |  $c=[c_1,c_2,c_3]$ |  $c \sim [\mathcal{N}_{[-1,1]}(c_2,1),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{c} \sim [\mathcal{\delta}(\tilde{c_1}-c_1),\mathcal{\delta}(\tilde{c_2}-c_2),\mathcal{\delta}(\tilde{c_3}-c_3)]$  | Shared between views, causal dependencies between $c_2 \rightarrow c_1$ |
| Style      |  $s=[s_1,s_2,s_3]$ |  $s \sim [\mathcal{U}([-1,1]),\mathcal{N}_{[-1,1]}(s_3,1),test-test-test-test$    |   $\tilde{s} \sim [\mathcal{N}_{[-1,1]}(s_1,1)$,$\mathcal{N}_{[-1,1]}(s_2,1)$,$\mathcal{N}_{[-1,1]}(s_3,1)]$               | Stochastic between views, causal dependencies between $c_2 \rightarrow s_3$, $s_3 \rightarrow s_2$ |
| View-Specific|$m=[m_1,m_2,m_3]$ |  $m \sim [\mathcal{U}([-1,1]),\mathcal{\delta}(0),\mathcal{\delta}(0)]$  |   $\tilde{m} \sim [\mathcal{\delta}(0),\mathcal{\delta}(0),\mathcal{U}([-1,1])]$  | $m_1$ is specific to _View 1_, $m_2$ is constant, $m_3$ is specific to _View 2_ |

$\mathcal{N}_{[-1,1]}$ refers to a normal distribution truncated to the $[-1,1]$ interval. 



## Download
- - -

The sample pairs and their associated ground-truth factors can be downloaded here:
* [_3DBoxIdent_ part 1]()
* [_3DBoxIdent_ part 2]()

The folder structure follows the following logic:

```
hues_positions_rotations               # example folder name
├── samples                            # sample pairs x = {x,\tilde{x}}
│   ├── m1                             # x, first elements of each sample pair (e.g., "000000.png")
│   │   └── *.png
│   └── m2                             # \tilde{x}, second elements of each sample pair (e.g., "000000.png")
│       └── *.png
└── factors                            # ground truth factors pairs z = {z,\tilde{z}} = {[c,s,m],[\tilde{c},\tilde{s},\tilde{m}]}
    ├── m1                             # z, first elements of each gt factors pair 
    │   ├── latents.npy                # z, distributed across blender support
    │   └── raw_latents.npy            # z, distributed across raw support
    └── m2                             # \tilde{z}, second elements of each gt factors pair 
        ├── latents.npy  
        └── raw_latents.npy   
```
Each folder contains a full dataset with specific <span style="color: RoyalBlue;">Block Type</span> associated with each <span style="color: Crimson;">Information Block</span>. The example `hues_positions_rotations` folder name follows the folder name template `content_style_view\_specific`. In this example the _Hues_ block is content information, _Positions_ is style information and _Rotations_ is view-specific information. 
## Custom Generation
- - -

## BibTeX
- - -
If you find our datasets useful, please cite our paper:

```bibtex
@article{daunhawer2023multimodal,
  author = {
    Daunhawer, Imant and
    Bizeul, Alice and
    Palumbo, Emanuele and
    Marx, Alexander and
    Vogt, Julia E.
  },
  title = {
    Identifiability Results for Multimodal Contrastive Learning
  },
  booktitle = {International Conference on Learning Representations},
  year = {2023}
}
```
<!-- TODO adjust bibtex? -->

## Acknowledgements
- - -
This project builds on the following resources. Please cite them appropriately.
- https://github.com/blender/blender
- https://github.com/facebookresearch/clevr-dataset-gen
- https://github.com/ysharma1126/ssl_identifiability
- https://github.com/brendel-group/cl-ica 
