# Image Remove Background Rembg (mtb)
## Documentation
- Class name: `Image Remove Background Rembg (mtb)`
- Category: `mtb/image`
- Output node: `False`

This node leverages the Rembg library to remove the background from images, optionally applying alpha matting for improved edge handling and allowing for post-processing of the mask. It supports customization of the background color for the resultant image.
## Input types
### Required
- **`image`**
    - The input image from which the background is to be removed. It plays a crucial role in determining the effectiveness of the background removal process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image`
- **`alpha_matting`**
    - A boolean flag indicating whether alpha matting should be applied to improve the quality of the edges in the output image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`alpha_matting_foreground_threshold`**
    - The threshold value for determining the foreground in the context of alpha matting, affecting the sharpness and clarity of the foreground edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`alpha_matting_background_threshold`**
    - The threshold value for determining the background in the context of alpha matting, influencing the separation between the foreground and the background.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`alpha_matting_erode_size`**
    - The size of the erosion applied in the alpha matting process, which can help in refining the edges of the foreground object.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`post_process_mask`**
    - A boolean flag indicating whether the mask generated during the background removal process should undergo post-processing for potentially improved quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`bgcolor`**
    - The color to be used for the background of the output image, allowing for customization of the resultant image's appearance.
    - Comfy dtype: `COLOR`
    - Python dtype: `Tuple[int, int, int, int]`
## Output types
- **`Image (rgba)`**
    - Comfy dtype: `IMAGE`
    - The image with the background removed, presented in RGBA format.
    - Python dtype: `torch.Tensor`
- **`Mask`**
    - Comfy dtype: `MASK`
    - The mask used for background removal.
    - Python dtype: `torch.Tensor`
- **`Image`**
    - Comfy dtype: `IMAGE`
    - The image with a new background color applied, presented in RGB format.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRemoveBackgroundRembg:
    """Removes the background from the input using Rembg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_matting": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "alpha_matting_foreground_threshold": (
                    "INT",
                    {"default": 240, "min": 0, "max": 255},
                ),
                "alpha_matting_background_threshold": (
                    "INT",
                    {"default": 10, "min": 0, "max": 255},
                ),
                "alpha_matting_erode_size": (
                    "INT",
                    {"default": 10, "min": 0, "max": 255},
                ),
                "post_process_mask": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "bgcolor": (
                    "COLOR",
                    {"default": "#000000"},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "Image (rgba)",
        "Mask",
        "Image",
    )
    FUNCTION = "remove_background"
    CATEGORY = "mtb/image"

    # bgcolor: Optional[Tuple[int, int, int, int]]
    def remove_background(
        self,
        image,
        alpha_matting,
        alpha_matting_foreground_threshold,
        alpha_matting_background_threshold,
        alpha_matting_erode_size,
        post_process_mask,
        bgcolor,
    ):
        pbar = comfy.utils.ProgressBar(image.size(0))
        images = tensor2pil(image)

        out_img = []
        out_mask = []
        out_img_on_bg = []

        for img in images:
            img_rm = remove(
                data=img,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                session=None,
                only_mask=False,
                post_process_mask=post_process_mask,
                bgcolor=None,
            )

            # extract the alpha to a new image
            mask = img_rm.getchannel(3)

            # add our bgcolor behind the image
            image_on_bg = Image.new("RGBA", img_rm.size, bgcolor)

            image_on_bg.paste(img_rm, mask=mask)

            image_on_bg = image_on_bg.convert("RGB")

            out_img.append(img_rm)
            out_mask.append(mask)
            out_img_on_bg.append(image_on_bg)

            pbar.update(1)

        return (pil2tensor(out_img), pil2tensor(out_mask), pil2tensor(out_img_on_bg))

```
