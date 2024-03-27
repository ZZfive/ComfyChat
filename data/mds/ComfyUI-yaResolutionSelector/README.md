# Yet Another Resolution Selector (YARS)

A slightly different Resolution Selector node, allowing to freely change base resolution and aspect ratio, with options to maintain the pixel count or use the base resolution as the highest or lowest dimension.

![Example](yeetctor.PNG)

---

## Installation

Exactly the same as with other simple custom nodes.

- Click the green **Code** button, select **Download Zip**, and unpack it in your ComfyUI `custom_nodes` directory

or

- Clone this repository by running `git clone https://github.com/Tropfchen/ComfyUI-yaResolutionSelector.git` in your ComfyUI `custom_nodes` directory

To uninstall:

- Delete `ComfyUI-yaResolutionSelector` in your ComfyUI custom_nodes directory

## Use

Simply right click on `Empty Latent Image` and choose one of `Prepend yaResolution Selector`, you can also find new nodes in `utils` menu.

#### yaResolution Selector (Advanced)

Recommend node with flexible ratio choice

- **base_resolution**: in theory, this should match the base resolution of the model you're using (512 for SD1.5, 1024 for SDXL).
- **Overextend**: By default, with a 2:1 ratio and a base resolution of 512, the node will output 256x512. However, when 'overextend' is set to true, the output will be 512x1024. This is beneficial when using models that can still generate good images at resolutions larger than the base resolution.
- **constant_resolution**: The node will attempt to output dimensions with the same pixel count as a 1:1 image generated with the base resolution. This outputs resolutions similar to those recommended for use with SDXL. It's also useful with node connected to an image rescale node, especially when you know the maximum resolution that your GPU VRAM can safely handle.

#### yaResolution Selector

Basic version for users that prefer aspect ratio presets. New presets can be added by modifying nodes.py file, remember to always include values in format width:height
