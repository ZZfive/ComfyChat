# Latent Scale by ratio
## Documentation
- Class name: `Latent Scale by ratio`
- Category: `Derfuu_Nodes/Modded nodes/Latent`
- Output node: `False`

This node scales a latent representation proportionally based on a specified modifier, using a chosen method for scaling and cropping. It's designed to adjust the dimensions of latent spaces while maintaining their aspect ratios, making it useful for modifying the size of generated content without distorting its original proportions.
## Input types
### Required
- **`latent`**
    - The latent representation to be scaled. It serves as the primary input for the scaling operation, determining the base structure that will be modified.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`modifier`**
    - A floating-point value that specifies the degree to which the latent representation is scaled. It plays a crucial role in determining the final size of the scaled latent space.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`scale_method`**
    - Specifies the method used for scaling the latent representation. It affects the quality and characteristics of the scaling operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`crop`**
    - Determines how the scaled latent representation is cropped, affecting the final dimensions and aspect ratio of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The scaled latent representation, adjusted according to the specified modifier, scale method, and cropping option.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LatentScale_Ratio:
    scale_methods = (["nearest-exact", "bilinear", "area"],)
    crop_methods = (["disabled", "center"],)

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "modifier": field.FLOAT,
                "scale_method": cls.scale_methods,
                "crop": cls.crop_methods,
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "scale"
    CATEGORY = TREE_LATENTS

    def scale(self, latent, scale_method, crop, modifier):

        size = sizes.get_latent_size(latent, True)

        lat_width = size[0]
        lat_height = size[1]
        #
        # width = latent["samples"][2]
        # height = latent["samples"][3]

        cls = latent.copy()
        cls["samples"] = comfy.utils.common_upscale(latent["samples"], lat_width // 8, lat_height // 8, scale_method, crop)
        return (cls,)

```
