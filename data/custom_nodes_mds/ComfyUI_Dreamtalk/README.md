# ComfyUI Dreamtalk (Unofficial Support)

Unofficial [Dreamtalk](https://github.com/ali-vilab/dreamtalk) support for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Important Updates
- **2024/03/29:** Added installation from ComfyUI Manager
- **2024/03/28:** Added ComfyUI nodes and workflow examples

## Basic Workflow
This [workflow](examples/dream_talk_simple.json) shows the basic usage on making an image into a talking face video.

 ![](examples/dream_talk_simple.jpg)

## Advanced Workflow
This [workflow](examples/dream_talk_advanced.json) shows the advanced usage on making the whole image talking, instead of just a cropped face.

 ![](examples/dream_talk_advanced.jpg)

## Installation
- Install from ComfyUI Manager (search for `dreamtalk`, make sure `ffmpeg` is installed)

- Download or git clone this repository into the ComfyUI/custom_nodes/ directory and run:
```
sudo apt install ffmpeg
pip install -r requirements.txt
```

## Download Checkpoints
Put the downloaded `denoising_network.pth` and `renderer.pt` into `checkpoints` folder. 

Above two files can be requested from the original author as follows:

> In light of the social impact, we have ceased public download access to checkpoints. If you want to obtain the checkpoints, please request it by emailing mayf18@mails.tsinghua.edu.cn . It is important to note that sending this email implies your consent to use the provided method **solely for academic research purposes**.

