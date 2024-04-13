# Get image size
## Documentation
- Class name: `Get image size`
- Category: `Derfuu_Nodes/Functions`
- Output node: `False`

This node is designed to calculate the dimensions of an image, returning the width and height as integers along with the original size tuple. It abstracts the complexity of handling image data structures to provide straightforward size information.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the image for which the size is to be determined. It is crucial for calculating the dimensions of the image, affecting the node's execution by determining the output size values.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - Represents the height of the image in pixels.
    - Python dtype: `int`
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - A tuple containing the width and height of the image, providing a compact representation of the image's dimensions.
    - Python dtype: `Tuple[int, int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImageSize:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "TUPLE",)
    CATEGORY = TREE_FUNCTIONS

    FUNCTION = 'get_size'

    def get_size(self, image):
        size = sizes.get_image_size(image)
        return (size[0], size[1], size, )

```
