# CausalMultiview3DIdent & Multiview3DIdent: Identifiability Benchmark Datasets

[[_Multiview3DIdent_]()][[_CausalMultiview3DIdent_]()][[Paper](https://openreview.net/forum?id=U_2kuqoTcB)]

Official code base for the generation of the _CausalMultiview3DIdent_ and _Multiview3DIdent_ datasets presented in the paper [Identifiability Results for Multimodal Contrastive
Learning](https://openreview.net/forum?id=U_2kuqoTcB) and featured at the [CLeaR datasets track 2023](https://www.cclear.cc/2023/CallforDatasets). Both datasets offer an identifiability benchmark by providing image/image pairs generated from controlled ground-truth factors using the [Blender](https://github.com/blender/blender) software. Ground-truth factors are independantly sampled for the _Multiview3DIdent_ dataset while non-trivial dependencies exist between  _CausalMultiview3DIdent_'s factors. 

<p align="center">
  <img src="https://github.com/alicebizeul/Multimodal3DIdent/blob/main/sample_image.png" alt="Multiviewl3DIdent dataset example images" width=570 />
</p>

This repository offers the possibility to adjust generative assumptions to fit personal preferences and generate custom-made datasets. 

## Dataset Description
- - -
The training/validation/test datasets consist of 250,000/10,000/10,000
samples of image pairs respectively. Images depict a colored teapot in front of a colored background, illuminated by a colored spotlight. _View 1_ displays the teaport with a **metallic** texture; _View 2_ displays the teapot with a **rubber** texture.

Each image is controlled by _9_ factors of variation partitioned into _3_ information blocks detailed below: 

| <span style="color: Crimson;">Information Block</span>      | Description | Raw Support | Blender Support | Details |
| ---------- | ----------- |-------- |-------- |-------- |
| <span style="color: FireBrick;">Rotation</span>   | $\alpha$-angle      |   $[-\pi;\pi]$|   $[-1;1]$   | Object $\alpha$ rotation angle|
| <span style="color: FireBrick;">Rotation</span>   | $\beta$-angle        |  $[-\pi;\pi]$|   $[-1;1]$   | Object $\beta$ rotation angle|
| <span style="color: FireBrick;">Rotation</span>   | $\gamma$-angle        | $[-\pi;\pi]$|   $[-1;1]$    | Spotlight rotation angle|
| <span style="color: DarkOrange;">Position</span>   | $x$-coordinate       | $[-1;1]$|    $[-2;2]$    | Object $x$-coordinate|
| <span style="color: DarkOrange;">Position</span>    | $y$-coordinate        | $[-1;1]$|    $[-2;2]$   | Object $y$-coordinate|
| <span style="color: DarkOrange;">Position</span>   | $z$-coordinate        | $[-1;1]$|   $[-2;2]$     | Object $z$-coordinate|
| <span style="color: Pink;">Hue</span>   | Object color      |         $[-1;1]$| $[-\pi; \pi]$ | Object HSV color, H parameter|
| <span style="color: Pink;">Hue</span>    | Background color       |  $[-1;1]$|   $[-\pi; \pi]$    | Background HSV color, H parameter |
| <span style="color: Pink;">Hue</span>    | Spotlight color      |  $[-1;1]$|   $[-\pi; \pi]$   | Spotlight HSV color, H parameter|

The generation of ground truth factors pairs follows specific distributions according to three sets of rules (i.e., <span style="color: RoyalBlue;">Block Type</span>). For each dataset, <span style="color: Crimson;">Information Blocks</span> are associated with a unique <span style="color: RoyalBlue;">Block Type</span>. Distributions linked to each <span style="color: RoyalBlue;">Block Type</span> are detailed below for _Multiview3DIdent_ and _CausalMultiview3DIdent_ respectively.

All combinaisons of Information Block - Block Type are available. 

### Multiview3DIdent

| <span style="color: RoyalBlue;">Block Type</span>  |Symbol | View 1 Distribution | View 2 Distribution | 
|----------------- |------------------- |------------------- |-------------------- |
| **<span style="color: MediumTurquoise;">Content</span>**           |  $c=[c_1,c_2,c_3]$ |  $c \sim [\mathcal{U}([-1,1]),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{c} \sim [\mathcal{\delta}(\tilde{c_1}-c_1),\mathcal{\delta}(\tilde{c_2}-c_2),\mathcal{\delta}(\tilde{c_3}-c_3)]$  | 
| **<span style="color: DodgerBlue ;">Style</span>**             |  $s=[s_1,s_2,s_3]$ |  $s \sim [\mathcal{U}([-1,1]),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{s} \sim [\mathcal{N}_{[-1,1]}(s_1,1),\mathcal{N}_{[-1,1]}(s_2,1),\mathcal{N}_{[-1,1]}(s_3,1)]$ 
| **<span style="color: LightSkyBlue;">View-Specific</span>**     |  $m=[m_1,m_2,m_3]$ |  $m \sim [\mathcal{U}([-1,1]),\mathcal{\delta}(0),\mathcal{\delta}(0)]$  |   $\tilde{m} \sim [\mathcal{\delta}(0),\mathcal{\delta}(0),\mathcal{U}([-1,1])]$  | 

### CausalMultiview3DIdent

| <span style="color: RoyalBlue;">Block Type</span> |             Symbol | View 1 Distribution              | View 2 Distribution                               | 
|----------------- |------------------- |----------------------------------|-------------------------------------------------- |
| **<span style="color: MediumTurquoise;">Content</span>**          |  $c=[c_1,c_2,c_3]$ |  $c \sim [\mathcal{N}_{[-1,1]}(c_2,1),\mathcal{U}([-1,1]),\mathcal{U}([-1,1])]$    |   $\tilde{c} \sim [\mathcal{\delta}(\tilde{c_1}-c_1),\mathcal{\delta}(\tilde{c_2}-c_2),\mathcal{\delta}(\tilde{c_3}-c_3)]$  | 
| **<span style="color: DodgerBlue ;">Style</span>**            |  $s=[s_1,s_2,s_3]$ |  $s \sim [\mathcal{U}([-1,1]),\mathcal{N}_{[-1,1]}(s_3,1),\mathcal{N}_{[-1,1]}(c_2,1)]$    |   $\tilde{s} \sim [\mathcal{N}_{[-1,1]}(s_1,1),\mathcal{N}_{[-1,1]}(s_2,1),\mathcal{N}_{[-1,1]}(s_3,1)]$               | 
| **<span style="color: LightSkyBlue;">View-Specific</span>**    |  $m=[m_1,m_2,m_3]$ |  $m \sim [\mathcal{U}([-1,1]),\mathcal{\delta}(0),\mathcal{\delta}(0)]$  |   $\tilde{m} \sim [\mathcal{\delta}(0),\mathcal{\delta}(0),\mathcal{U}([-1,1])]$  | 

## Download
- - -

The sample pairs and their associated ground-truth factors can be downloaded here:
* [_Multiview3DIdent_]()
* [_CausalMultiview3DIdent_]()

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