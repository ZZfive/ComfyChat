# Crop (mtb)
## Documentation
- Class name: `Crop (mtb)`
- Category: `mtb/crop`
- Output node: `False`

The Crop node is designed to extract a specific region from an image or a set of images based on provided coordinates or a bounding box. It supports optional masking to refine the cropping process, and can handle both individual images and batches of images, adjusting the output accordingly.
## Input types
### Required
- **`image`**
    - The image or batch of images to be cropped. This is the primary input for the cropping operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image(s) to specify or refine the area to be cropped. If provided, the mask is used to adjust the cropping boundaries.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
- **`x`**
    - The x-coordinate of the top-left corner of the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The y-coordinate of the top-left corner of the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The width of the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox`**
    - An optional bounding box specifying the cropping area. If provided, it overrides the individual x, y, width, and height parameters, serving as a more direct method to define the crop area.
    - Comfy dtype: `BBOX`
    - Python dtype: `Optional[Tuple[int, int, int, int]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The cropped image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The optionally cropped mask, if a mask was provided as input.
    - Python dtype: `Optional[torch.Tensor]`
- **`bbox`**
    - Comfy dtype: `BBOX`
    - The cropping data including coordinates (x, y) and dimensions (width, height) of the cropped area.
    - Python dtype: `Tuple[int, int, int, int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Crop:
    """Crops an image and an optional mask to a given bounding box

    The bounding box can be given as a tuple of (x, y, width, height) or as a BBOX type
    The BBOX input takes precedence over the tuple input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "x": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "width": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "bbox": ("BBOX",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOX")
    FUNCTION = "do_crop"

    CATEGORY = "mtb/crop"

    def do_crop(
        self, image: torch.Tensor, mask=None, x=0, y=0, width=256, height=256, bbox=None
    ):
        image = image.numpy()
        if mask is not None:
            mask = mask.numpy()

        if bbox is not None:
            x, y, width, height = bbox

        cropped_image = image[:, y : y + height, x : x + width, :]
        cropped_mask = None
        if mask is not None:
            cropped_mask = (
                mask[:, y : y + height, x : x + width] if mask is not None else None
            )
        crop_data = (x, y, width, height)

        return (
            torch.from_numpy(cropped_image),
            torch.from_numpy(cropped_mask) if cropped_mask is not None else None,
            crop_data,
        )

```
