# Image Flip
## Documentation
- Class name: `JWImageFlip`
- Category: `jamesWalker55`
- Output node: `False`

The JWImageFlip node is designed for flipping images either horizontally or vertically based on the specified direction. This operation can be essential for data augmentation or correcting image orientation.
## Input types
### Required
- **`image`**
    - The image tensor to be flipped. This is the primary input for the flipping operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`direction`**
    - Specifies the direction of the flip, either 'horizontal' or 'vertical'. This determines how the image will be manipulated.
    - Comfy dtype: `['horizontal', 'vertical']`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The flipped image tensor, altered according to the specified direction.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
@register_node("JWImageLoadRGB", "Image Load RGB")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path": ("STRING", {"default": "./image.png"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, path: str):
        assert isinstance(path, str)

        img = load_image(path)
        return (img,)

```
