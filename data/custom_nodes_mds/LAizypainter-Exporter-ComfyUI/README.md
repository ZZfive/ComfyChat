# LAizypainter-Exporter-ComfyUI

This exporter is a plugin for [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/), which can export tasks for [LAizypainter](https://github.com/DimaChaichan/LAizypainter). 

LAizypainter is a Photoshop plugin with which you can send tasks directly to a Stable Diffusion server.
More information about a [Task](https://github.com/DimaChaichan/LAizypainter?tab=readme-ov-file#task)

## Installation

To install, clone this repository into `ComfyUI/custom_nodes` folder with `git clone https://github.com/DimaChaichan/LAizypainter-Exporter-ComfyUI` and restart ComfyUI.

## How to use
Click with the right mouse button outside a node and select "**LAizyPainter Export...**". 
![open_dialog.gif](assets%2Fopen_dialog.gif)

There are three tabs which represent the keys to be set in the LZY file. Config, Variable, Prompt
![overview.png](assets%2Foverview.png)

#### Config
Set the Config for the task. [More Info](https://github.com/DimaChaichan/LAizypainter?tab=readme-ov-file#task)

#### Variable
Set the variables for th Task [More Info](https://github.com/DimaChaichan/LAizypainter/blob/main/doc%2Fvariables.md)

#### Prompt
Connect the Variables with the Prompt.
