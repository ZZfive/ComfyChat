# Get Image Size
## Documentation
- Class name: `Get Image Size`
- Category: `Masquerade Nodes`
- Output node: `False`

This node is designed to determine the dimensions of an image, specifically its width and height. It abstracts the process of extracting size information from an image, providing a straightforward way to obtain these critical attributes.
## Input types
### Required
- **`image`**
    - The 'image' parameter is crucial as it represents the image from which the size is to be determined. The node uses this image to extract its width and height, which are essential for various image processing tasks.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The output consists of two integers representing the width and height of the image, respectively. These dimensions are fundamental for understanding the size and aspect ratio of the image.
    - Python dtype: `Tuple[int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - UltimateSDUpscale



## Source code
```python
class GetImageSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT","INT",)
    FUNCTION = "get_size"

    CATEGORY = "Masquerade Nodes"

    def get_size(self, image):
        image_size = image.size()
        image_width = int(image_size[2])
        image_height = int(image_size[1])
        return (image_width, image_height,)

```
