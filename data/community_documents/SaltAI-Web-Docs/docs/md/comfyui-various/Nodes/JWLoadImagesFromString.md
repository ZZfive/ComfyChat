# Load Images From String
## Documentation
- Class name: `JWLoadImagesFromString`
- Category: `jamesWalker55`
- Output node: `False`

This node is designed to load images directly from a string input, facilitating the conversion of textual image data into a usable image format for further processing or visualization.
## Input types
### Required
- **`paths`**
    - Specifies the textual path or string representation of the image location, serving as the source from which the image is loaded.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`ignore_missing_images`**
    - Determines whether to ignore images that cannot be found at the specified paths, allowing for graceful handling of missing files.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is the image tensor loaded from the specified string path or the default image tensor if the path is invalid.
    - Python dtype: `torch.Tensor`
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
