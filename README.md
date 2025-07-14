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
1. Clone this repo:
```bash
git clone https://github.com/seunghyuns98/VideoColorGrading.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```
3. Download 

### Inference

Run inference code on our provided demo videos. \
Make sure to change directory of pretrained model to the path you download pretrained weights in configs/prompts/video_demo.yaml file.

```bash
python video_demo.py \
--ref_path examples/reference.png \ #PATH To Your reference images or videos
--input_path examples/video1.mp4 \ #PATH To Your Input Video 
--save_path output/example1.mp4 \ #PATH To Your Output Folder
# --config configs/prompts/video_demo.yaml
# --seed 42
# --size 512
# --steps 25
```

