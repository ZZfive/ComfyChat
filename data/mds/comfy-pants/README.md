# comfy-pants: custom nodes for ComfyUI
This repo contains a number of QoL nodes I made for myself that I figured I may as well make available for everyone else.
## Nodes
- Make Square Node
  - Will pad an image to make it square with a chosen filling strategy. Sort of like "Resize and fill" from a1111.
- TextEncodeAIO
  - Tried to hack StylePile-like (the A1111 extension, shoutout btw) behavior into a ClipTextEncode node, but I'm not sure I'm even doing it right.
## Installation
First clone the repo to your `custom_nodes` folder:

    cd path/to/ComfyUI/custom_nodes
    git clone https://github.com/pants007/comfy-pants.git

Then enter the repo folder and install dependencies through pip:

    cd comfy_pants
    pip install -r requirements.txt
