# Latent Scale to side
## Documentation
- Class name: `Latent Scale to side`
- Category: `Derfuu_Nodes/Modded nodes/Latent`
- Output node: `False`

This node is designed to proportionally scale a latent representation to fit a specified side length, adjusting its dimensions while maintaining the aspect ratio. It utilizes a specified scaling and cropping method to achieve the desired dimensions.
## Input types
### Required
- **`latent`**
    - The latent representation to be scaled. It's the core input for the scaling operation, determining the base dimensions and content that will be adjusted.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`side_length`**
    - The target length for the specified side (width or height) of the latent representation, guiding the scaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`side`**
    - Specifies which side (width or height) the side_length applies to, or if it should apply to the longest side.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale_method`**
    - The method used for scaling the latent representation, affecting the quality and approach of the scaling.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`crop`**
    - Determines how the latent representation is cropped during the scaling process, affecting the final composition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The scaled latent representation, adjusted to fit the specified side length while maintaining its aspect ratio.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LatentScale_Side:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "side_length": field.INT,
                "side": (["Longest", "Width", "Height"],),
                "scale_method": (cls.upscale_methods,),
                "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = TREE_LATENTS

    def upscale(self, latent, side_length: int, side: str, scale_method, crop):

        size = sizes.get_latent_size(latent, True)

        lat_width = size[0]
        lat_height = size[1]

        width = lat_width
        height = lat_height

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = lat_height / lat_width
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = lat_width / lat_height
                height = side_length
                width = width_ratio * height
            return width, height

        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)


        width = math.ceil(width)
        height = math.ceil(height)

        cls = latent.copy()
        cls["samples"] = comfy.utils.common_upscale(latent["samples"], width // 8, height // 8, scale_method, crop)
        return (cls,)

```
