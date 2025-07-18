# <p align=center> **Video Color Grading via Look-Up Table Generation** </p>
<div align="center">
<img src='assets/teaser.png' style="height:250px"></img>
<br>

Seunghyun Shin<sup>1</sup>, 
Dongmin Shin<sup>2</sup>, 
Jisu Shin<sup>1 </sup>, 
Hae-Gon Jeon<sup>2 &dagger;</sup>, 
Joon-Young Lee<sup>3 &dagger;</sup>, 

<br>
<sup>1</sup>GIST
<sup>2</sup>Yonsei University
<sup>3</sup>Adobe Research

ICCV 2025

 <a href='https://arxiv.org'><img src='https://img.shields.io/badge/arXiv-2504.01016-b31b1b.svg'></a> &nbsp;
</div>

## 📝 Introduction

We present a reference-based video color grading framework. Our key idea is to generate a look-up table (LUT) for color attribute alignment between reference scenes and input video via a diffusion model.

If you find Video Color Grading useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

- `[26/06/2025]` 🎉🎉🎉 Video Color Grading is accepted by ICCV 2025.


## 🚀 Quick Start

### Installation
1. Clone the repository:
```bash
git clone https://github.com/seunghyuns98/VideoColorGrading.git
```
2. Install dependencies:
- Directly Generate conda with our bash script: 
```bash
source fast_env.sh
```
- Or Install manually(please refer to [fast_env.sh](fast_env.sh))

3. Download the pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1NpWXjQxo6ZdOVdSoCzVhCik58WMXXqqC?usp=sharing) and place them in the 'pretrained/' directory.

### Inference

Run inference code using the provided example reference image and input video. \
If you placed pretrained models in a directory other than 'pretrained/', make sure to update its path in configs/prompts/video_demo.yaml.

```bash
python video_demo.py \
--ref_path examples/reference.jpg \
--input_path examples/video1.mp4 \
--save_path output/example1.mp4 
```

## 🏋️‍♂️ Train Your Own Model

Training consists of two steps:
1. Training GS-Extractor
2. Training L-Diffuser 

Before training, make sure to update the config files with your environment:
```yaml
pretrained_model_path : PATH To Your Pretrained Stable-Diffusion-Model
clip_model_path: PATH To Your Pretrained CLIP-Model
step1_checkpoint_path: PATH To Your Pretrained Step1 Model
etc.
```
You can see config files at confgis folder 

Furthermore, change your lut path in your dataloader

### 📁 Dataset Preparation
We use the Condensed Movie Dataset which consists of over 33,000 clips from 3,000 movies covering the salient parts of the films and has two-minutes running time for each clip in average. \
100 LUT bases which are selected as distinctive LUTs from the $400$ LUTs of the Video Harmonization Dataset. \
You can download them through: [Google Drive](https://drive.google.com/file/d/1iHljoQGH1OGNC-yWGX8XUcR6hip8_4zt/view?usp=sharing) \
They are originally from: [Condensed Movie Dataset](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies) & [Video Harmonization Dataset](https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube)

### 🔧 Training Phase 1

```commandline
torchrun --nnodes=1 --nproc_per_node=8 train_step1.py --config configs/training/train_stage_1.yaml
```

### 🔧 Training Phase 2

```commandline
torchrun --nnodes=1 --nproc_per_node=8 train_step2.py --config configs/training/train_stage_2.yaml
```

### 📊 Evaluation

Coming Soon

## 🤝 Contributing

- Welcome to open issues and pull requests.
- Welcome to optimize the inference speed and memory usage, e.g., through model quantization, distillation, or other acceleration techniques.

## ❤️ Acknowledgement

We have used codes from other great research work, including [Animate-Anyone](https://github.com/guoqincode/Open-AnimateAnyone), [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter). We sincerely thank the authors for their awesome work!

## 📜 Citation

If you find this work helpful, please consider citing:

```BibTeXw
@article{preparing,
  title={Video Color Grading via Look-Up Table Generation},
  author={Shin, Seuunghyun and Shin, Dongmin and Shin, Jisu and Jeon, Hae-Gon and Lee, Joon-Young},
  journal={arXiv preprint arXiv:2504.01016},
  year={2025}
}
```
