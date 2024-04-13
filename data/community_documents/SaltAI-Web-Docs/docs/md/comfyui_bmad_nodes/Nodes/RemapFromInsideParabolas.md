# RemapFromInsideParabolas
## Documentation
- Class name: `RemapFromInsideParabolas`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to perform a remapping operation from the perspective of inside two parabolas. It utilizes a source mask defined by two parabolas to transform the spatial layout of an image, adjusting its dimensions according to specified width and height parameters. The transformation aims to modify the image's appearance by altering its geometry in a way that reflects the curvature and orientation of the parabolas.
## Input types
### Required
- **`src_mask_with_2_parabolas`**
    - Specifies the source mask that contains two parabolas. This mask is used as a reference for the remapping operation, dictating the transformation's geometric basis.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Determines the width of the output image after the remapping operation. This parameter allows for the adjustment of the image's horizontal dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the output image post-remapping. This parameter enables the modification of the image's vertical dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The result of the remapping operation, which is an image transformed according to the curvature and orientation of the two parabolas defined in the source mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapFromInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "src_mask_with_2_parabolas": ("MASK",),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    def send_remap(self, src_mask_with_2_parabolas, width, height):
        from .utils.remaps import remap_from_inside_parabolas
        return ({
                    "func": remap_from_inside_parabolas,
                    "xargs": [tensor2opencv(src_mask_with_2_parabolas, 1), width, height],
                    "dims": (width, height)
                },)

```
