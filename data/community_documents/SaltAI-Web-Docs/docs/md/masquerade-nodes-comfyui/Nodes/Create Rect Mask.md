# Create Rect Mask
## Documentation
- Class name: `Create Rect Mask`
- Category: `Masquerade Nodes`
- Output node: `False`

The Create Rect Mask node is designed to generate rectangular masks based on specified dimensions and positions. It supports both absolute and percentage-based dimensions, allowing for flexible mask creation relative to the image size. Additionally, it accommodates different origin points to adjust the mask's position according to the image's coordinate system.
## Input types
### Required
- **`mode`**
    - Specifies the mode of dimension specification, either as absolute values or percentages of the image size, influencing how the mask's dimensions are calculated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`origin`**
    - Determines the origin point for the mask's position, affecting how the mask's coordinates are adjusted relative to the image's coordinate system.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x`**
    - The x-coordinate of the mask's starting point, which is adjusted based on the specified origin and mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`y`**
    - The y-coordinate of the mask's starting point, similar to the x-coordinate, it is adjusted based on the origin and mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`width`**
    - The width of the mask, which can be specified in absolute values or as a percentage of the image width, depending on the mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`height`**
    - The height of the mask, which can be specified in absolute values or as a percentage of the image height, depending on the mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_width`**
    - The width of the image, used to calculate the mask's dimensions when the mode is set to percentage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_height`**
    - The height of the image, used to calculate the mask's dimensions when the mode is set to percentage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`copy_image_size`**
    - An optional parameter that, if provided, uses the dimensions of the specified image to override the image_width and image_height parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated rectangular mask, represented as a binary tensor with ones indicating the mask area and zeros elsewhere.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateRectMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["percent", "pixels"],),
                "origin": (["topleft", "bottomleft", "topright", "bottomright"],),
                "x": ("FLOAT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "y": ("FLOAT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "width": ("FLOAT", {"default": 50, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "height": ("FLOAT", {"default": 50, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "image_width": ("INT", {"default": 512, "min": 64, "max": VERY_BIG_SIZE, "step": 64}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": VERY_BIG_SIZE, "step": 64}),
            },
            "optional": {
                "copy_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_mask"

    CATEGORY = "Masquerade Nodes"

    def create_mask(self, mode, origin, x, y, width, height, image_width, image_height, copy_image_size = None):
        min_x = x
        min_y = y

        max_x = min_x + width
        max_y = min_y + height
        if copy_image_size is not None:
            size = copy_image_size.size()
            image_width = size[2]
            image_height = size[1]
        if mode == "percent":
            min_x = min_x / 100.0 * image_width
            max_x = max_x / 100.0 * image_width
            min_y = min_y / 100.0 * image_height
            max_y = max_y / 100.0 * image_height

        if origin == "bottomleft" or origin == "bottomright":
            min_y, max_y = image_height - max_y, image_height - min_y
        if origin == "topright" or origin == "bottomright":
            min_x, max_x = image_width - max_x, image_width - min_x
            
        mask = torch.zeros((image_height, image_width))
        mask[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1] = 1
        return (mask.unsqueeze(0),)

```
