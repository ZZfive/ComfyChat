# Latent Mirror node for ComfyUI

Node to mirror a latent along the Y (vertical / left to right) or X (horizontal / top to bottom) axis. Best used as part of an img2img workflow:

Example nodes JSON: https://github.com/spro/comfyui-mirror/blob/main/examples/latent-mirror-example.json (using [Efficiency Nodes](https://github.com/LucianoCirino/efficiency-nodes-comfyui))

![example nodes](https://github.com/spro/comfyui-latentmirror/blob/main/examples/nodes.png?raw=true)

Example images (after an Img2Img step):

![example results](https://github.com/spro/comfyui-latentmirror/blob/main/examples/example.png?raw=true)

## Installation

Clone the repo into `ComfyUI/custom_nodes`...

```
cd ComfyUI/custom_nodes
git clone https://github.com/spro/comfyui-latentmirror
```

... and restart and refresh ComfyUI.
