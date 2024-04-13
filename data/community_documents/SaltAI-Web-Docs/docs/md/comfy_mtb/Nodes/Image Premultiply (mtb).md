# Image Premultiply (mtb)
## Documentation
- Class name: `Image Premultiply (mtb)`
- Category: `mtb/image`
- Output node: `False`

The Image Premultiply node is designed to blend an image with a mask, optionally inverting the mask, to produce premultiplied images. This process is essential for compositing tasks where the transparency and blending of images need to be controlled.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the source image to be premultiplied with the mask. It plays a crucial role in determining the final appearance of the output image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The 'mask' parameter is used to define the areas of the image that will be affected during the premultiplication process. It is essential for achieving the desired transparency and blending effects.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert`**
    - The 'invert' parameter allows for the inversion of the mask before premultiplication, offering flexibility in how the mask is applied to the image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`RGBA`**
    - Comfy dtype: `IMAGE`
    - The output is a premultiplied image with the mask applied, potentially inverted, resulting in an RGBA image where transparency and blending have been adjusted.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImagePremultiply:
    """Premultiply image with mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "mtb/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGBA",)
    FUNCTION = "premultiply"

    def premultiply(self, image, mask, invert):
        images = tensor2pil(image)
        masks = tensor2pil(mask) if invert else tensor2pil(1.0 - mask)
        single = len(mask) == 1
        masks = [x.convert("L") for x in masks]

        out = []
        for i, img in enumerate(images):
            cur_mask = masks[0] if single else masks[i]

            img.putalpha(cur_mask)
            out.append(img)

        # if invert:
        #     image = Image.composite(image,Image.new("RGBA", image.size, color=(0,0,0,0)), mask)
        # else:
        #     image = Image.composite(Image.new("RGBA", image.size, color=(0,0,0,0)), image, mask)

        return (pil2tensor(out),)

```
