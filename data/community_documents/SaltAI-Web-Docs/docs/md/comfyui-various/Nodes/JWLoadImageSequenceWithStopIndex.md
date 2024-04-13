# Batch Load Image Sequence With Stop Index
## Documentation
- Class name: `JWLoadImageSequenceWithStopIndex`
- Category: `jamesWalker55`
- Output node: `False`

This node is designed to batch load a sequence of images based on a specified start and stop index, with an option to include or exclude the stop index image. It also provides the capability to ignore missing images, ensuring robustness in handling incomplete image sequences.
## Input types
### Required
- **`path_pattern`**
    - A string pattern representing the path to the images, where an index placeholder is used to load a sequence of images.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_index`**
    - The starting index from which to begin loading the image sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`stop_index`**
    - The index at which to stop loading the image sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`inclusive`**
    - A boolean flag indicating whether the stop index image should be included in the loaded sequence.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
- **`ignore_missing_images`**
    - A boolean flag that, when set to true, allows the node to ignore missing images and proceed with loading available images.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The loaded sequence of images as a tensor.
    - Python dtype: `torch.Tensor`
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
