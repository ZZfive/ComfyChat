# This is an implementation of MotionCtrl for ComfyUI

[MotionCtrl](https://github.com/TencentARC/MotionCtrl): A Unified and Flexible Motion Controller for Video Generation 

## Install

1. Clone this repo into custom_nodes directory of ComfyUI location

2. Run pip install -r requirements.txt

3. Download the weights of MotionCtrl  [motionctrl.pth](https://huggingface.co/TencentARC/MotionCtrl/blob/main/motionctrl.pth) and put it to `ComfyUI/models/checkpoints`

## Nodes

Four nodes `Load Motionctrl Checkpoint` & `Motionctrl Cond` & `Motionctrl Sample Simple` & `Load Motion Camera Preset` & `Load Motion Traj Preset` & `Select Image Indices` &`Motionctrl Sample`

## Tools

[Motion Traj Tool](https://chaojie.github.io/ComfyUI-MotionCtrl/tools/draw.html) Generate motion trajectories

<img src="assets/traj.png" raw=true>

[Motion Camera Tool](https://chaojie.github.io/ComfyUI-MotionCtrl/tools/index.html) Generate motion camera points

<img src="assets/camera.png" raw=true>

## Examples

base workflow

<img src="assets/base_wf.png" raw=true>

https://github.com/chaojie/ComfyUI-MotionCtrl/blob/main/workflow_motionctrl_base.json

<video controls autoplay="true">
    <source 
   src="assets/dog.mp4" 
   type="video/mp4" 
  />
</video>

unofficial implementation "MotionCtrl deployed on AnimateDiff" workflow:

<img src="assets/scribble_wf.png" raw=true>

https://github.com/chaojie/ComfyUI-MotionCtrl/blob/main/workflow_motionctrl.json

1. Generate LVDM/VideoCrafter Video
2. Select Images->Scribble
3. Use AnimateDiff Scribble SparseCtrl
