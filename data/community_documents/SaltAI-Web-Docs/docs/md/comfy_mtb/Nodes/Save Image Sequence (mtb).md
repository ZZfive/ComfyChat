# Save Image Sequence (mtb)
## Documentation
- Class name: `Save Image Sequence (mtb)`
- Category: `mtb/IO`
- Output node: `True`

The Save Image Sequence node is designed to save a sequence of images to a specified directory, allowing for the inclusion of additional metadata and customization of the file naming convention. It facilitates the organization and storage of generated image sequences, making it easier to manage and retrieve specific frames or sequences.
## Input types
### Required
- **`images`**
    - A list of images to be saved. This parameter is crucial as it directly influences the node's primary function of saving image sequences.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`filename_prefix`**
    - A prefix for the filename, used to customize the naming convention of the saved images. It helps in categorizing and identifying images within a sequence.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`current_frame`**
    - The index of the current frame in the sequence. It is used to determine the naming of the saved image file, facilitating the organization of the image sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`ui`**
    - A UI component displaying the saved images, including filenames, subfolder names, and the type of output. It provides a visual representation and quick access to the saved images.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveImageSequence:
    """Save an image sequence to a folder. The current frame is used to determine which image to save.

    This is merely a wrapper around the `save_images` function with formatting for the output folder and filename.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Sequence"}),
                "current_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999999},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "mtb/IO"

    def save_images(
        self,
        images,
        filename_prefix="Sequence",
        current_frame=0,
        prompt=None,
        extra_pnginfo=None,
    ):
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        # results = list()
        # for image in images:
        #     i = 255. * image.cpu().numpy()
        #     img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        #     metadata = PngInfo()
        #     if prompt is not None:
        #         metadata.add_text("prompt", json.dumps(prompt))
        #     if extra_pnginfo is not None:
        #         for x in extra_pnginfo:
        #             metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        #     file = f"{filename}_{counter:05}_.png"
        #     img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        #     results.append({
        #         "filename": file,
        #         "subfolder": subfolder,
        #         "type": self.type
        #     })
        #     counter += 1

        if len(images) > 1:
            raise ValueError("Can only save one image at a time")

        resolved_path = Path(self.output_dir) / filename_prefix
        resolved_path.mkdir(parents=True, exist_ok=True)

        resolved_img = (
            resolved_path / f"{filename_prefix}_{current_frame:05}.png"
        )

        output_image = images[0].cpu().numpy()
        img = Image.fromarray(
            np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
        )
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        img.save(resolved_img, pnginfo=metadata, compress_level=4)
        return {
            "ui": {
                "images": [
                    {
                        "filename": resolved_img.name,
                        "subfolder": resolved_path.name,
                        "type": self.type,
                    }
                ]
            }
        }

```
