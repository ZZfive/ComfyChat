# ImageConcatenate
## Documentation
- Class name: `ImageConcanate`
- Category: `KJNodes`
- Output node: `False`

The ImageConcanate node is designed to concatenate two images either horizontally or vertically, based on the specified direction. It enables the creation of composite images by seamlessly joining individual images together.
## Input types
### Required
- **`image1`**
    - The first image to be concatenated. It serves as the base image to which the second image will be joined.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be concatenated. It is joined to the first image in the specified direction.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`direction`**
    - Specifies the direction in which the second image will be concatenated to the first. Options include 'right', 'down', 'left', and 'up', with 'right' being the default.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The concatenated image as a result of joining the two input images in the specified direction.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "KJNodes"

    def concanate(self, image1, image2, direction):
        if direction == 'right':
            row = torch.cat((image1, image2), dim=2)
        elif direction == 'down':
            row = torch.cat((image1, image2), dim=1)
        elif direction == 'left':
            row = torch.cat((image2, image1), dim=2)
        elif direction == 'up':
            row = torch.cat((image2, image1), dim=1)
        return (row,)

```
