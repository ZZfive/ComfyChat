# Batch Load Image Sequence
## Documentation
- Class name: `JWLoadImageSequence`
- Category: `jamesWalker55`
- Output node: `False`

This node is designed for batch loading a sequence of images, facilitating the processing or analysis of multiple images in a collective manner. It streamlines the workflow by allowing for the efficient handling of image sequences as a batch, enhancing productivity and computational efficiency.
## Input types
### Required
- **`path_pattern`**
    - Specifies the pattern or path used to identify and load the sequence of images. This parameter is essential for locating the images to be processed as a batch.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_index`**
    - Defines the starting index from which images in the sequence begin to be loaded, allowing for selective processing of the sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_count`**
    - Determines the number of frames or images to be loaded from the sequence, starting from the start_index. This allows for control over the extent of the batch processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ignore_missing_images`**
    - Indicates whether to ignore missing images within the specified range, enabling uninterrupted batch processing despite potential gaps in the sequence.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images loaded as a sequence. This facilitates further processing or analysis of the images collectively, rather than individually.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
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
