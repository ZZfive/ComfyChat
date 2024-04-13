# Batch Save Image Sequence
## Documentation
- Class name: `JWSaveImageSequence`
- Category: `jamesWalker55`
- Output node: `True`

This node is designed for batch saving a sequence of images to a specified path, with options for formatting the file names, starting index, and handling file overwrites. It facilitates organized output management in image processing workflows.
## Input types
### Required
- **`images`**
    - The sequence of images to be saved. This is the primary data that the node operates on, determining the content of the output files.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`path_pattern`**
    - A string pattern that defines the file naming convention and path for saving the images, which can include placeholders for dynamic naming based on indices.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_index`**
    - The starting index for naming the saved image files, allowing for sequential ordering based on this initial value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`overwrite`**
    - A flag indicating whether existing files at the target path should be overwritten. If set to 'false', new non-conflicting paths will be generated.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
@register_node("JWLoadImageSequence", "Batch Load Image Sequence")
class _:
    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "path_pattern": (
                "STRING",
                {"default": "./frame{:06d}.png", "multiline": False},
            ),
            "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            "frame_count": ("INT", {"default": 16, "min": 1, "step": 1}),
            "ignore_missing_images": (("false", "true"), {"default": "false"}),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        path_pattern: str,
        start_index: int,
        frame_count: int,
        ignore_missing_images: str,
    ):
        ignore_missing_images: bool = ignore_missing_images == "true"

        # generate image paths to load
        image_paths = []
        for i in range(start_index, start_index + frame_count):
            try:
                image_paths.append(path_pattern.format(i))
            except KeyError:
                image_paths.append(path_pattern.format(i=i))

        if ignore_missing_images:
            # remove missing images
            image_paths = [p for p in image_paths if os.path.exists(p)]
        else:
            # early check for missing images
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image does not exist: {path}")

        if len(image_paths) == 0:
            raise RuntimeError("Image sequence empty - no images to load")

        imgs = []
        for path in image_paths:
            img = load_image(path)
            # img.shape => torch.Size([1, 768, 768, 3])
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)

        return (imgs,)

```
