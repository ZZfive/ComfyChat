# ResolutionSelector for ComfyUI

A custom node for Stable Diffusion [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to enable easy selection of image resolutions for SDXL SD15 SD21

- Select base SDXL resolution, width and height are returned as `INT` values which can be connected to latent image inputs or other inputs such as the `CLIPTextEncodeSDXL` `width, height, target_width, target_height`. 
- Resolution list based off what is currently being used in the [Fooocus SDXL Web UI](https://github.com/lllyasviel/Fooocus).
- If using older models such as SD 1.5 or SD 2.1 use the `base_adjustment` dropdown. This will reduce the returned width and height values to suit the selected model whilst maintaining the image aspect ratio.

```terminal
# Example
SDXL base_resolution       1024x1024
SD21 adjustment returns    768x768
SD15 adjustment returns    512x512  
```


### Installation

```
# Change to the directory you installed ComfyUI
cd pathTo/ComfyUI

# Change to the custom_nodes directory ie.
cd custom_nodes
```

```terminal
# Clone the repo into custom_nodes
git clone https://github.com/bradsec/ComfyUI_ResolutionSelector.git

# Restart ComfyUI
```

### Usage after install
`Add Node > utils > Resolution Selector`  
  
![node_example](resolution_selector_node.png)

#### SDXL hookup example
![sdxl_hookup](sdxl_hookup.png)

#### SD15 hookup example
![sd15_hookup](sd15_hookup.png)