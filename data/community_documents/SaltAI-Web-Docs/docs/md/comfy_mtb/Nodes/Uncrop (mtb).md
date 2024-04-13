# Uncrop (mtb)
## Documentation
- Class name: `Uncrop (mtb)`
- Category: `mtb/crop`
- Output node: `False`

The Uncrop node is designed to reverse the effects of a cropping operation on an image, restoring it to its original dimensions by utilizing bounding box information. It aims to seamlessly integrate the cropped portion back into the larger image context, adjusting for factors like border blending to ensure a natural appearance.
## Input types
### Required
- **`image`**
    - The original image that was partially cropped. This image serves as the base for the uncropping operation, ensuring that the cropped portion can be accurately reinserted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`crop_image`**
    - The image that was cropped out and is to be reinserted into the original image. It is essential for the uncropping process to know what part of the image is being reintegrated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`bbox`**
    - The bounding box specifying the region of the image that was cropped. This information is crucial for determining where to place the cropped portion back into the original image.
    - Comfy dtype: `BBOX`
    - Python dtype: `Tuple[int, int, int, int]`
- **`border_blending`**
    - A parameter controlling the blending of the borders between the cropped portion and the original image. It affects the smoothness of the transition, aiming for a seamless integration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of the uncropping operation, which is the modified image with the cropped portion reinserted.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Uncrop:
    """Uncrops an image to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "border_blending": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_crop"

    CATEGORY = "mtb/crop"

    def do_crop(self, image, crop_image, bbox, border_blending):
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle(
                (0, 0, width - 1, height - 1), outline=border_color, width=border_width
            )
            return bordered_image

        single = image.size(0) == 1
        if image.size(0) != crop_image.size(0):
            if not single:
                raise ValueError(
                    "The Image batch count is greater than 1, but doesn't match the crop_image batch count. If using batches they should either match or only crop_image must be greater than 1"
                )

        images = tensor2pil(image)
        crop_imgs = tensor2pil(crop_image)
        out_images = []
        for i, crop in enumerate(crop_imgs):
            if single:
                img = images[0]
            else:
                img = images[i]

            # uncrop the image based on the bounding box
            bb_x, bb_y, bb_width, bb_height = bbox

            paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
            # log.debug(f"Paste region: {paste_region}")
            # new_region = adjust_paste_region(img.size, paste_region)
            # log.debug(f"Adjusted paste region: {new_region}")
            # # Check if the adjusted paste region is different from the original

            crop_img = crop.convert("RGB")

            log.debug(f"Crop image size: {crop_img.size}")
            log.debug(f"Image size: {img.size}")

            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)

            blend = img.convert("RGBA")
            mask = Image.new("L", img.size, 0)

            mask_block = Image.new("L", (bb_width, bb_height), 255)
            mask_block = inset_border(mask_block, int(blend_ratio / 2), (0))

            mask.paste(mask_block, paste_region)
            log.debug(f"Blend size: {blend.size} | kind {blend.mode}")
            log.debug(f"Crop image size: {crop_img.size} | kind {crop_img.mode}")
            log.debug(f"BBox: {paste_region}")
            blend.paste(crop_img, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            blend.putalpha(mask)
            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)

```
