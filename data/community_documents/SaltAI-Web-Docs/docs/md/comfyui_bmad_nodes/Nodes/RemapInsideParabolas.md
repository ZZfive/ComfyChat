# RemapInsideParabolas
## Documentation
- Class name: `RemapInsideParabolas`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to perform a remapping operation based on the geometry of two parabolas within a given mask. It adjusts the spatial layout of an image or a mask to align with the contours defined by these parabolas, potentially altering the image's perspective or distortion to fit within the new bounds.
## Input types
### Required
- **`dst_mask_with_2_parabolas`**
    - Specifies the destination mask that contains two parabolas. This mask is used to determine the new spatial layout for the remapping operation.
    - Comfy dtype: `MASK`
    - Python dtype: `numpy.ndarray`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output includes the remapped coordinates and potentially the bounding box of the region of interest, adjusted according to the parabolas' geometry.
    - Python dtype: `Dict[str, numpy.ndarray]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas):
        from .utils.remaps import remap_inside_parabolas_simple
        return ({
                    "func": remap_inside_parabolas_simple,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1)],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)

```
