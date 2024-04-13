# Image Save To Path
## Documentation
- Class name: `JWImageSaveToPath`
- Category: `jamesWalker55`
- Output node: `True`

This node is designed to save an image to a specified file path, allowing for the inclusion of additional metadata such as prompts and other PNG-specific information.
## Input types
### Required
- **`path`**
    - The file path where the image will be saved. This path determines the location and filename of the output image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`image`**
    - The image tensor to be saved. It must be a 3-channel image tensor.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`overwrite`**
    - A boolean flag indicating whether an existing file at the specified path should be overwritten.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
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
