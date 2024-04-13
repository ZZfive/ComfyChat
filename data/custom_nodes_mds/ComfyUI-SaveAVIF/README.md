# ComfyUI-SaveAVIF
A custom node on ComfyUI that saves images in AVIF format.

Workflow can be loaded from images saved at this node.

## Description:

It can be used in the same way as the Save Image node.

Note that encoding to AVIF takes a little time.

c_quality: Set quality for color (0 - 100, where 100 is lossless)

enc_speed: Encoder speed (0 - 10, slowest/smallest - fastest/largest)

When encoding is finished, the node outputs an image, which is a pass-through of the input.
This is intended for use such as notifying the end of encoding by connecting a node that plays a sound.

## Installation: 

Install it using ComfyUI Manager.

Or use `git clone https://github.com/pkpkTech/ComfyUI-SaveAVIF` in your ComfyUI custom nodes directory

and `pip install -r requirements.txt`

