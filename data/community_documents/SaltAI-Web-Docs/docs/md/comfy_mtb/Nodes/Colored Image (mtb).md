# Colored Image (mtb)
## Documentation
- Class name: `Colored Image (mtb)`
- Category: `mtb/generate`
- Output node: `False`

The Colored Image node is designed to generate constant color images of specified sizes. It allows for the creation of simple, uniform color backgrounds that can be optionally overlaid with a foreground image and mask, providing a straightforward way to produce images with a solid color base.
## Input types
### Required
- **`color`**
    - Specifies the color of the generated image. This parameter is crucial for defining the visual appearance of the output.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`width`**
    - Determines the width of the generated image. This parameter allows for customization of the image size to fit specific requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated image, enabling the user to specify the desired dimensions of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`foreground_image`**
    - An optional parameter that allows for the overlay of a foreground image onto the solid color background, adding complexity to the generated image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`foreground_mask`**
    - When a foreground image is provided, this optional parameter can be used to specify a mask that determines how the foreground image blends with the background color.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Produces an image with the specified solid color, dimensions, and optional foreground overlay.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColoredImage:
    """Constant color image of given size."""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("COLOR",),
                "width": ("INT", {"default": 512, "min": 16, "max": 8160}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8160}),
            },
            "optional": {
                "foreground_image": ("IMAGE",),
                "foreground_mask": ("MASK",),
            },
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_img"

    def resize_and_crop(self, img, target_size):
        # Calculate scaling factors for both dimensions
        scale_x = target_size[0] / img.width
        scale_y = target_size[1] / img.height

        # Use the smaller scaling factor to maintain aspect ratio
        scale = max(scale_x, scale_y)

        # Resize the image based on calculated scale
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.ANTIALIAS)

        # Calculate cropping coordinates
        left = (img.width - target_size[0]) / 2
        top = (img.height - target_size[1]) / 2
        right = (img.width + target_size[0]) / 2
        bottom = (img.height + target_size[1]) / 2

        # Crop and return the image
        return img.crop((left, top, right, bottom))

    def resize_and_crop_thumbnails(self, img, target_size):
        img.thumbnail(target_size, Image.ANTIALIAS)
        left = (img.width - target_size[0]) / 2
        top = (img.height - target_size[1]) / 2
        right = (img.width + target_size[0]) / 2
        bottom = (img.height + target_size[1]) / 2
        return img.crop((left, top, right, bottom))

    def render_img(
        self,
        color,
        width,
        height,
        foreground_image: Optional[torch.Tensor] = None,
        foreground_mask: Optional[torch.Tensor] = None,
    ):
        image = Image.new("RGBA", (width, height), color=color)
        output = []
        if foreground_image is not None:
            fg_images = tensor2pil(foreground_image)
            fg_masks = [None] * len(
                fg_images
            )  # Default to None for each foreground image

            if foreground_mask is not None:
                if foreground_image.size()[0] != foreground_mask.size()[0]:
                    raise ValueError(
                        "Foreground image and mask must have same batch size"
                    )
                fg_masks = tensor2pil(foreground_mask)

            for fg_image, fg_mask in zip(fg_images, fg_masks):
                # Resize and crop if dimensions mismatch
                if fg_image.size != image.size:
                    fg_image = self.resize_and_crop(fg_image, image.size)
                    if fg_mask:
                        fg_mask = self.resize_and_crop(fg_mask, image.size)

                if fg_mask:
                    output.append(
                        Image.composite(
                            fg_image.convert("RGBA"),
                            image,
                            fg_mask,
                        ).convert("RGB")
                    )
                else:
                    if fg_image.mode != "RGBA":
                        raise ValueError(
                            "Foreground image must be in 'RGBA' mode when no mask is provided, got {fg_image.mode}"
                        )
                    output.append(
                        Image.alpha_composite(image, fg_image).convert("RGB")
                    )

        elif foreground_mask is not None:
            log.warn("Mask ignored because no foreground image is given")

        output = pil2tensor(output)

        return (output,)

```
