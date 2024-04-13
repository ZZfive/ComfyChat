# PixelTiledKSampleUpscalerProvider
## Documentation
- Class name: `PixelTiledKSampleUpscalerProvider`
- Category: `ImpactPack/Upscale`
- Output node: `False`

The PixelTiledKSampleUpscalerProvider node is designed to facilitate the upscaling of images through a tiled k-sample approach. It leverages a specific sampling technique to enhance image resolution in a segmented manner, ensuring detailed and high-quality upscaling.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for scaling the image, impacting the overall upscaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model`**
    - The model used for upscaling, determining the quality and characteristics of the output image.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`vae`**
    - Variational Autoencoder (VAE) used in conjunction with the model to improve image quality through latent space manipulation.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`seed`**
    - Seed for random number generation, ensuring reproducibility of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the upscaling process, affecting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the upscaling process, allowing customization of the upscaling behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Name of the sampler used in the upscaling process, influencing the texture and details of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Scheduler used to adjust the upscaling process dynamically, optimizing the output quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - Positive keywords to guide the upscaling process, enhancing specific aspects of the image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative keywords to counterbalance the positive ones, preventing undesired elements in the upscaled image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`denoise`**
    - Denoising level applied to the upscaled image, improving clarity and reducing noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tile_width`**
    - Width of the tiles used in the upscaling process, determining the segmentation of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - Height of the tiles used in the upscaling process, determining the segmentation of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiling_strategy`**
    - Strategy for tiling the image during the upscaling process, affecting the overall approach to image segmentation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`upscale_model_opt`**
    - Optional upscale model options, providing additional customization for the upscaling process.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `str`
- **`pk_hook_opt`**
    - Optional post-kernel hook options, allowing for further customization after the upscaling kernel is applied.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `str`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - The upscaler object created by the node, capable of performing the tiled k-sample upscaling on images.
    - Python dtype: `core.PixelTiledKSampleUpscaler`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PixelTiledKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt=None, pk_hook_opt=None):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_size=max(tile_width, tile_height))
            return (upscaler, )
        else:
            print("[ERROR] PixelTiledKSampleUpscalerProvider: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")

```
