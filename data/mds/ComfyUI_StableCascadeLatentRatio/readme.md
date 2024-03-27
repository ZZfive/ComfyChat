![Untitled](https://github.com/Guillaume-Fgt/ComfyUI_StableCascadeLatentRatio/assets/66461774/38d8dd2b-87c2-4f92-93a7-15a2b740ee25)

A ComfyUI custom node to create empty latents for Stable Cascade:

- purple background color at creation
- width and height incrementation of 64 by default
- possibility to lock the aspect ratio. Changing the width or the height will update the other dimension accordingly
- switch width/height at execution (not displayed in the node, just taken into account at run time)
- in order to be able to use Latent From Batch node, stage_c and stage_b batch sizes are separated into two widgets

To install, simply git clone the repo in `ComfyUI/custom_nodes` folder:
```
git clone https://github.com/Guillaume-Fgt/ComfyUI_StableCascadeLatentRatio.git
```

You will find the node under `latent/Stable Cascade Latent Ratio`.
