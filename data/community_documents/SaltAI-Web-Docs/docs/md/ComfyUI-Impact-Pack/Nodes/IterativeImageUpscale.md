# Iterative Upscale (Image)
## Documentation
- Class name: `IterativeImageUpscale`
- Category: `ImpactPack/Upscale`
- Output node: `False`

The IterativeImageUpscale node is designed to progressively upscale images through a series of steps, enhancing the resolution and detail of the input image with each iteration. This node aims to achieve higher-quality upscaling by applying the upscale operation multiple times, each time increasing the image's dimensions towards a desired scale factor.
## Input types
### Required
- **`pixels`**
    - The input pixels to be upscaled, typically representing images. This parameter is crucial for initiating the iterative upscaling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`upscale_factor`**
    - Specifies the target upscale factor by which the image's dimensions should be increased. This factor determines the final size of the upscaled image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Defines the number of iterative steps to perform during the upscaling process. Each step incrementally increases the image's size, contributing to the gradual enhancement of image quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`temp_prefix`**
    - An optional prefix for temporary files generated during the upscaling process. This can be used for debugging or intermediate result inspection.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`upscaler`**
    - The upscaling model or method to be used for each upscaling step. This parameter is central to the upscaling process, determining the approach for increasing image resolution.
    - Comfy dtype: `UPSCALER`
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - The variational autoencoder used in the upscaling process, particularly relevant when working with latent representations of images.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output of the IterativeImageUpscale node is the upscaled image, enhanced through the iterative upscaling process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ImageBatch](../../Comfy/Nodes/ImageBatch.md)



## Source code
```python
class IterativeImageUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "pixels": ("IMAGE", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",),
                     "vae": ("VAE",),
                },
                "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, pixels, upscale_factor, steps, temp_prefix, upscaler, vae, unique_id):
        if temp_prefix == "":
            temp_prefix = None

        core.update_node_status(unique_id, "VAEEncode (first)", 0)
        if upscaler.is_tiled:
            latent = nodes.VAEEncodeTiled().encode(vae, pixels, upscaler.tile_size)[0]
        else:
            latent = nodes.VAEEncode().encode(vae, pixels)[0]

        refined_latent = IterativeLatentUpscale().doit(latent, upscale_factor, steps, temp_prefix, upscaler, unique_id)

        core.update_node_status(unique_id, "VAEDecode (final)", 1.0)
        if upscaler.is_tiled:
            pixels = nodes.VAEDecodeTiled().decode(vae, refined_latent[0], upscaler.tile_size)[0]
        else:
            pixels = nodes.VAEDecode().decode(vae, refined_latent[0])[0]

        core.update_node_status(unique_id, "", None)

        return (pixels, )

```
