# ComfyUI aichemy nodes
Simple node to handle scaling of YOLOv8 segmentation masks

## Installation
Download or git clone this repository inside `ComfyUI/custom_nodes/` directory or use the Manager.

1. Git clone this repo to the `ComfyUI/custom_nodes/` path or use the Manager.

   `git clone https://github.com/HAL41/ComfyUI_aichemy_nodes`


## Nodes
### YOLOv8 Segmentation
While the returned annotated image from YOLOv8 has proper scaling, the returned mask does not. The segmentation is done on a lower resolution and with padding. The mask doesn't align properly when you try to simply resize the mask to the original resolution. 

This simple node does the computation to remove the padding and resize the mask to the original resolution. This way you can quickly compose images together using the mask found by YOLOv8 model.

The comparison can be seen on the following image. The image on the left shows the properly scaled annotated images straight from YOLOv8. The top row shows the bad composite image created by scaled mask from standard YOLOv8 node. The bottom row shows the same images using this custom node. As you can see the border above shoulders is gone as well as the early cropping on the bottom of the image.
![YOLOv8-workflow](imgs/YOLOv8-workflow.png)

The two images can be compared here:
![YOLOv8-comparison](imgs/YOLOv8-comparison.png)