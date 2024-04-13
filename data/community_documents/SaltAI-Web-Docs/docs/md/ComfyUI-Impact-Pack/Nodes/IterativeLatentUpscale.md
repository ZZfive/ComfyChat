# Iterative Upscale (Latent/on Pixel Space)
## Documentation
- Class name: `IterativeLatentUpscale`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node is designed to iteratively upscale latent representations of images over a specified number of steps, allowing for gradual and controlled image enlargement. It leverages an upscaling method to progressively increase the resolution of the latent space representation, enhancing the detail and quality of the generated images with each iteration.
## Input types
### Required
- **`samples`**
    - The initial latent representation of the image to be upscaled. It serves as the starting point for the iterative upscaling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`upscale_factor`**
    - The total factor by which the image's dimensions are to be increased by the end of the iterative process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - The number of iterations to divide the upscaling process into, allowing for gradual enlargement and refinement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`temp_prefix`**
    - An optional prefix for temporary file saving during the upscaling process, facilitating intermediate result inspection.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`upscaler`**
    - The specific upscaling method or model to be used for each upscaling step.
    - Comfy dtype: `UPSCALER`
    - Python dtype: `torch.nn.Module`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The upscaled latent representation after the iterative upscaling process.
    - Python dtype: `Dict[str, torch.Tensor]`
- **`vae`**
    - Comfy dtype: `VAE`
    - The variational autoencoder used during the upscaling process.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)



## Source code
```python
class IterativeLatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "samples": ("LATENT", ),
                     "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1, "max": 10000, "step": 0.1}),
                     "steps": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                     "temp_prefix": ("STRING", {"default": ""}),
                     "upscaler": ("UPSCALER",)
                },
                "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("latent", "vae")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, samples, upscale_factor, steps, temp_prefix, upscaler, unique_id):
        w = samples['samples'].shape[3]*8  # image width
        h = samples['samples'].shape[2]*8  # image height

        if temp_prefix == "":
            temp_prefix = None

        upscale_factor_unit = max(0, (upscale_factor-1.0)/steps)
        current_latent = samples
        scale = 1

        for i in range(steps-1):
            scale += upscale_factor_unit
            new_w = w*scale
            new_h = h*scale
            core.update_node_status(unique_id, f"{i+1}/{steps} steps | x{scale:.2f}", (i+1)/steps)
            print(f"IterativeLatentUpscale[{i+1}/{steps}]: {new_w:.1f}x{new_h:.1f} (scale:{scale:.2f}) ")
            step_info = i, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        if scale < upscale_factor:
            new_w = w*upscale_factor
            new_h = h*upscale_factor
            core.update_node_status(unique_id, f"Final step | x{upscale_factor:.2f}", 1.0)
            print(f"IterativeLatentUpscale[Final]: {new_w:.1f}x{new_h:.1f} (scale:{upscale_factor:.2f}) ")
            step_info = steps, steps
            current_latent = upscaler.upscale_shape(step_info, current_latent, new_w, new_h, temp_prefix)

        core.update_node_status(unique_id, "", None)

        return (current_latent, upscaler.vae)

```
