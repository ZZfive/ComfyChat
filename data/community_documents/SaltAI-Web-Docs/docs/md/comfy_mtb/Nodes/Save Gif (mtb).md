# Save Gif (mtb)
## Documentation
- Class name: `Save Gif (mtb)`
- Category: `mtb/IO`
- Output node: `True`

The Save Gif node is designed to convert a batch of images into a single GIF file, allowing for various customization options such as frame rate, size, optimization, and the addition of a ping-pong effect. This functionality is particularly useful for creating dynamic visual content from static images.
## Input types
### Required
- **`image`**
    - The primary input consisting of a batch of images to be converted into a GIF. Its role is crucial as it directly influences the content and appearance of the resulting GIF.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`fps`**
    - Specifies the frame rate of the GIF, affecting how smoothly the animation plays.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resize_by`**
    - A scaling factor for resizing the images, impacting the final dimensions of the GIF.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`optimize`**
    - A boolean flag indicating whether to apply optimization techniques to reduce the GIF file size without significantly compromising quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`pingpong`**
    - When enabled, adds a ping-pong effect to the animation, making it play forwards and then backwards.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`resample_filter`**
    - Optional. Specifies the resampling filter to use when resizing images, affecting the quality of the resized images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Optional[str]`
## Output types
- **`ui`**
    - Returns a dictionary containing the path to the generated GIF, facilitating its retrieval and display in user interfaces.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveGif:
    """Save the images from the batch as a GIF"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "resize_by": ("FLOAT", {"default": 1.0, "min": 0.1}),
                "optimize": ("BOOLEAN", {"default": False}),
                "pingpong": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "resample_filter": (list(PIL_FILTER_MAP.keys()),),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "mtb/IO"
    FUNCTION = "save_gif"

    def save_gif(
        self,
        image,
        fps=12,
        resize_by=1.0,
        optimize=False,
        pingpong=False,
        resample_filter=None,
    ):
        if image.size(0) == 0:
            return ("",)

        if resample_filter is not None:
            resample_filter = PIL_FILTER_MAP.get(resample_filter)

        pil_images = prepare_animated_batch(
            image,
            pingpong,
            resize_by,
            resample_filter,
        )

        ruuid = uuid.uuid4()
        ruuid = ruuid.hex[:10]
        out_path = f"{folder_paths.output_directory}/{ruuid}.gif"

        # Create the GIF from PIL images
        pil_images[0].save(
            out_path,
            save_all=True,
            append_images=pil_images[1:],
            optimize=optimize,
            duration=int(1000 / fps),
            loop=0,
        )

        results = [{"filename": f"{ruuid}.gif", "subfolder": "", "type": "output"}]
        return {"ui": {"gif": results}}

```
