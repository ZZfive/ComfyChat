
# trNodes

custom node modules for ComfyUI

## Installation

1. git clone this repo under `ComfyUI/custom_nodes`

2. install dependencies:

```bash
# go to project python folder
cd python_embeded
./python.exe -m pip install opencv-python scikit-image blendmodes
```

## Nodes

image_layering:
- Adds 1-3 layers of image on top of one another; will remove white background and use it as transparent layers
- TODO: add alpha control

color_correction:
- Adjusts the color of the target image according to another image; ported from stable diffusion WebUI

model_router:
- Batch reroutes `model`, `clip`, `vae`, `condition_1` and `condition_2` for cleaner workflow

## Use
1. Install nodes
2. open ComfyUI
3. You'll find the new nodes under `trNodes` folder


## External Nodes

[WAS Node suite](https://civitai.com/models/20793/was-node-suites-comfyui) 
- Image Blend by Mask: Blend two images by a mask (but all nodes are very good)

and even look this: https://civitai.com/models/24869/comfyui-custom-nodes-by-xss
