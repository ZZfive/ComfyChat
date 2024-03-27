# This is an implementation of ComfyUI MotionCtrl for SVD

[MotionCtrl for SVD](https://github.com/TencentARC/MotionCtrl/tree/svd)

## Install

1. Clone this repo into custom_nodes directory of ComfyUI location

2. Run pip install -r requirements.txt

3. Download the weights of MotionCtrl for SVD [motionctrl_svd.ckpt](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl_svd.ckpt) and put it to `ComfyUI/models/checkpoints`

## Examples

base workflow

<img src="assets/base_wf.png" raw=true>

https://github.com/chaojie/ComfyUI-MotionCtrl-SVD/blob/main/workflow.json

A little exploration workflow: for videos with relatively static camera angles, applying the same MotionCtrl to each frame, then combining images from corresponding positions to create a new video. The nth frame image is taken from the generated nth frame and merged again.

https://github.com/chaojie/ComfyUI-MotionCtrl-SVD/blob/main/workflow_video.json

original video: https://github.com/chaojie/ComfyUI-MotionCtrl-SVD/blob/main/assets/original.mp4
generate video: https://github.com/chaojie/ComfyUI-MotionCtrl-SVD/blob/main/assets/svd.mp4