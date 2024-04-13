# Constant Mask
## Documentation
- Class name: `Constant Mask`
- Category: `Masquerade Nodes`
- Output node: `False`

The Constant Mask node generates a mask of a specified size with a uniform value across all its elements. It can either use explicit dimensions or copy the dimensions from an existing image, providing flexibility in mask creation.
## Input types
### Required
- **`value`**
    - Specifies the uniform value to fill the mask with, affecting the mask's overall appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`explicit_height`**
    - Defines the explicit height of the mask. If set to 0 and no image size is copied, a default small size is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`explicit_width`**
    - Defines the explicit width of the mask. If set to 0 and no image size is copied, a default small size is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`copy_image_size`**
    - Optional. If provided, the mask dimensions are set to match this image's size, overriding explicit dimensions.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated mask as a tensor with the specified uniform value across its dimensions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ConstantMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.01}),
                "explicit_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "explicit_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
            },
            "optional": {
                "copy_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constant_mask"

    CATEGORY = "Masquerade Nodes"

    def constant_mask(self, value, explicit_height, explicit_width, copy_image_size = None):
        height = explicit_height
        width = explicit_width
        if copy_image_size is not None:
            size = copy_image_size.size()
            height = size[1]
            width = size[2]
        elif explicit_height == 0 or explicit_width == 0:
            # We'll just make a tiny mask and let it get resized by nodes further downstream
            height = 16
            width = 16

        result = torch.zeros([1, height, width])
        result[:,:,:] = value
        return (result,)

```
