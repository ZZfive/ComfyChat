# ComfyUI_InterpolateEverything
Custom nodes for interpolating between, well, everything in the Stable Diffusion ComfyUI.

Current functionality: 

* **Interpolate Poses** (preprocessors->pose->Interpolate Poses): Create a preprocessed ControlNet OpenPose input midway between two input images.

Future features:

* **Interpolate Lineart**: Use motion interpolation to create line-art ControlNet inputs midway between two input images.
* More?

To install:

**First, install https://github.com/Fannovel16/comfyui_controlnet_aux** if you haven't already:
```sh
cd <ComfyUI installation directory>/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux
cd comfyui_controlnet_aux
./install
```

Most of InterpolateEverything's (current) functionality depends on `comfyui_controlnet_aux`, and InterpolateEverything *should* fail to load if it's not installed.  

Next, install this repo in `custom_nodes` as well:

```sh
cd .. # Assuming you're still in comfyui_controlnet_aux; otherwise go to <ComfyUI installation directory>/custom_nodes
git clone https://github.com/shockz0rz/ComfyUI_InterpolateEverything.git
```

And you're done! 
