# PixelTiledKSampleUpscalerProviderPipe
## Documentation
- Class name: `PixelTiledKSampleUpscalerProviderPipe`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node is designed to upscale images using a tiled K-sample upscaling method. It leverages a specific upscaling model and various parameters to enhance image resolution in a tiled manner, ensuring efficient processing of large images by breaking them down into smaller, manageable tiles.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for scaling the image, impacting the upscaling quality and technique.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - Determines the random seed used for generating the upscaled image, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to perform during the upscaling process, affecting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configures the conditioning factor, influencing the upscaling process by adjusting the image's details and textures.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the sampler to be used, affecting the upscaling technique and the final image quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the upscaling process, impacting the progression and quality of the upscaling.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Determines the level of denoising applied to the upscaled image, affecting the clarity and smoothness of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tile_width`**
    - Sets the width of the tiles used in the upscaling process, influencing the processing efficiency and detail level in specific image areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - Sets the height of the tiles used in the upscaling process, influencing the processing efficiency and detail level in specific image areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiling_strategy`**
    - Defines the strategy for tiling the image during the upscaling process, affecting the overall efficiency and quality of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`basic_pipe`**
    - Provides the basic pipeline components required for the upscaling process, including models and configurations.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
### Optional
- **`upscale_model_opt`**
    - Optional parameter for specifying additional options for the upscaling model, allowing for further customization of the upscaling process.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `dict`
- **`pk_hook_opt`**
    - Optional parameter for providing hook options, enabling customization of the processing pipeline for specific upscaling needs.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `dict`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - The upscaled image produced by the node, showcasing enhanced resolution and detail.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE",)
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, tiling_strategy, basic_pipe, upscale_model_opt=None, pk_hook_opt=None):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            model, _, vae, positive, negative = basic_pipe
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_size=max(tile_width, tile_height))
            return (upscaler, )
        else:
            print("[ERROR] PixelTiledKSampleUpscalerProviderPipe: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")

```
