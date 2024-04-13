# Mask To Image (mtb)
## Documentation
- Class name: `Mask To Image (mtb)`
- Category: `mtb/generate`
- Output node: `False`

The Mask To Image node is designed to transform a mask into an RGB image, allowing for the visualization of masks by converting them into a more interpretable format.
## Input types
### Required
- **`mask`**
    - The mask input is crucial for the operation as it represents the data that will be converted into an RGB image. This transformation enables the visualization of the mask in a format that is easier to interpret and analyze.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`color`**
    - The color input specifies the color to be applied to the non-transparent parts of the mask, playing a significant role in the visual outcome of the RGB image.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`background`**
    - The background input determines the background color of the resulting image, allowing for customization of the visual context behind the mask.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an RGB image generated from the input mask, providing a visual representation of the mask data.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskToImage:
    """Converts a mask (alpha) to an RGB image with a color and background"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "color": ("COLOR",),
                "background": ("COLOR", {"default": "#000000"}),
            }
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "render_mask"

    def render_mask(self, mask, color, background):
        masks = tensor2np(mask)
        images = []
        for m in masks:
            _mask = Image.fromarray(m).convert("L")

            log.debug(
                f"Converted mask to PIL Image format, size: {_mask.size}"
            )

            image = Image.new("RGBA", _mask.size, color=color)
            # apply the mask
            image = Image.composite(
                image, Image.new("RGBA", _mask.size, color=background), _mask
            )

            # image = ImageChops.multiply(image, mask)
            # apply over background
            # image = Image.alpha_composite(Image.new("RGBA", image.size, color=background), image)

            images.append(image.convert("RGB"))

        return (pil2tensor(images),)

```
