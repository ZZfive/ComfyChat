# Bbox From Mask (mtb)
## Documentation
- Class name: `Bbox From Mask (mtb)`
- Category: `mtb/crop`
- Output node: `False`

This node is designed to extract a bounding box from a given mask, optionally inverting the mask and considering an image for size compatibility. It is capable of handling batch operations and ensures the bounding box accurately represents the area of interest within the mask or image.
## Input types
### Required
- **`mask`**
    - The mask tensor from which the bounding box is to be extracted. It plays a crucial role in determining the area of interest for cropping or analysis.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert`**
    - A boolean flag indicating whether the mask should be inverted before extracting the bounding box. This affects the area considered as the foreground for bounding box calculation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`image`**
    - An optional image tensor that, if provided, is used to ensure size compatibility with the mask. It affects the validation of mask and image sizes for batch operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`bbox`**
    - Comfy dtype: `BBOX`
    - The calculated bounding box, represented as a tuple of minimum x, minimum y, width, and height.
    - Python dtype: `Tuple[int, int, int, int]`
- **`image (optional)`**
    - Comfy dtype: `IMAGE`
    - The optional cropped image tensor, returned only if an image tensor was provided as input. This output is optional and may not always be present, depending on whether an image input was provided.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BboxFromMask:
    """From a mask extract the bounding box"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "BBOX",
        "IMAGE",
    )
    RETURN_NAMES = (
        "bbox",
        "image (optional)",
    )
    FUNCTION = "extract_bounding_box"
    CATEGORY = "mtb/crop"

    def extract_bounding_box(self, mask: torch.Tensor, invert: bool, image=None):
        # if image != None:
        #     if mask.size(0) != image.size(0):
        #         if mask.size(0) != 1:
        #             log.error(
        #                 f"Batch count mismatch for mask and image, it can either be 1 mask for X images, or X masks for X images (mask: {mask.shape} | image: {image.shape})"
        #             )

        #             raise Exception(
        #                 f"Batch count mismatch for mask and image, it can either be 1 mask for X images, or X masks for X images (mask: {mask.shape} | image: {image.shape})"
        #             )

        # we invert it
        _mask = tensor2pil(1.0 - mask)[0] if invert else tensor2pil(mask)[0]
        alpha_channel = np.array(_mask)

        non_zero_indices = np.nonzero(alpha_channel)

        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

        # Create a bounding box tuple
        if image != None:
            # Convert the image to a NumPy array
            imgs = tensor2np(image)
            out = []
            for img in imgs:
                # Crop the image from the bounding box
                img = img[min_y:max_y, min_x:max_x, :]
                log.debug(f"Cropped image to shape {img.shape}")
                out.append(img)

            image = np2tensor(out)
            log.debug(f"Cropped images shape: {image.shape}")
        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        return (
            bounding_box,
            image,
        )

```
