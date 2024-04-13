# Image scale to side
## Documentation
- Class name: `Image scale to side`
- Category: `Derfuu_Nodes/Modded nodes/Image`
- Output node: `False`

This node proportionally scales an image to fit a specified side length, either width or height, while maintaining the image's aspect ratio. It supports different methods for scaling and cropping, allowing for flexible image resizing operations.
## Input types
### Required
- **`image`**
    - The image to be scaled. It is the primary input for the scaling operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`side_length`**
    - The target length for the specified side (width or height) of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`side`**
    - Determines which side ('Width', 'Height', or 'Longest') of the image to scale to the target side_length.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_method`**
    - Specifies the method used for scaling the image, such as 'nearest-exact', 'bilinear', or 'area'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`crop`**
    - Indicates whether and how to crop the image after scaling, with options like 'disabled' or 'center'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The scaled image, returned as a tensor.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - Reroute
    - [ImpactImageInfo](../../ComfyUI-Impact-Pack/Nodes/ImpactImageInfo.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)



## Source code
```python
class ImageScale_Side:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side_length": field.INT,
                "side": (["Longest", "Width", "Height"],),
                "upscale_method": (cls.upscale_methods,),
                "crop": (cls.crop_methods,)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = TREE_IMAGES

    def upscale(self, image, upscale_method, side_length: int, side: str, crop):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)

        width_B = int(size[0])
        height_B = int(size[1])

        width = width_B
        height = height_B

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = height_B / width_B
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
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

        cls = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        cls = cls.movedim(1, -1)
        return (cls,)

```
