# Image scale by ratio
## Documentation
- Class name: `Image scale by ratio`
- Category: `Derfuu_Nodes/Modded nodes/Image`
- Output node: `False`

This node proportionally scales an image based on a specified ratio, adjusting its dimensions while maintaining the original aspect ratio. It supports various methods for upscale processing and optional cropping to fine-tune the output image's appearance.
## Input types
### Required
- **`image`**
    - The image to be scaled. The scaling process adjusts the image's dimensions while preserving its aspect ratio.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`upscale_by`**
    - The factor by which the image's dimensions should be scaled. A value greater than 1 enlarges the image, while a value less than 1 reduces its size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`upscale_method`**
    - Specifies the method used for scaling the image. Options include 'nearest-exact', 'bilinear', and 'area', each offering different trade-offs in terms of quality and computational cost.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`crop`**
    - Determines if and how the scaled image should be cropped. Options include 'disabled' (no cropping) and 'center' (crop to center), allowing for additional control over the final image appearance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The scaled image, with dimensions adjusted according to the specified ratio and method, optionally cropped as per the provided settings.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageScale_Ratio:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_by": field.FLOAT,
                "upscale_method": (cls.upscale_methods,),
                "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = TREE_IMAGES

    def upscale(self, image, upscale_method, upscale_by, crop):
        size = get_image_size(image)

        width_B = int(size[0])
        height_B = int(size[1])

        samples = image.movedim(-1, 1)

        height = math.ceil(height_B * upscale_by)
        width = math.ceil(width_B * upscale_by)
        cls = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        cls = cls.movedim(1, -1)
        return (cls,)

```
