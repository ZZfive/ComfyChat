# 🔧 Image Flip
## Documentation
- Class name: `ImageFlip+`
- Category: `essentials`
- Output node: `False`

The ImageFlip+ node provides functionality for flipping images along specified axes. It can flip images horizontally, vertically, or both, allowing for versatile image manipulation.
## Input types
### Required
- **`image`**
    - The image to be flipped. This parameter is crucial as it specifies the target image for the flipping operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`axis`**
    - Specifies the axis or axes along which the image will be flipped. Can be 'x' for horizontal, 'y' for vertical, or 'xy' for both.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[str]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The flipped image. This output is the result of flipping the input image along the specified axis or axes.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "axis": (["x", "y", "xy"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        image = torch.flip(image, dim)

        return(image,)

```
