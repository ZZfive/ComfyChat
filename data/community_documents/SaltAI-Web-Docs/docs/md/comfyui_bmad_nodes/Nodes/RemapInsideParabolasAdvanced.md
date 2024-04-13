# RemapInsideParabolasAdvanced
## Documentation
- Class name: `RemapInsideParabolasAdvanced`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed for advanced remapping of images based on the geometry of two parabolas within a destination mask. It allows for fine-tuned adjustments in both curve-wise and orthogonal directions, and offers the option to flip the orthogonal adjustment, enabling a high degree of control over the remapping process.
## Input types
### Required
- **`dst_mask_with_2_parabolas`**
    - A mask image containing two parabolas. This mask is used as a reference for the remapping process, dictating the geometric constraints within which the image is transformed.
    - Comfy dtype: `MASK`
    - Python dtype: `numpy.ndarray`
- **`curve_wise_adjust`**
    - A floating-point value that adjusts the intensity of the curve-wise (along the parabolas) remapping. This allows for nuanced control over how the image is stretched or compressed along the curves.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ortho_wise_adjust`**
    - A floating-point value that adjusts the intensity of the orthogonal (perpendicular to the parabolas) remapping. This enables precise control over the image's distortion in the direction orthogonal to the curves.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_ortho`**
    - A boolean flag that, when set to True, flips the direction of the orthogonal adjustment. This can be used to invert the direction of distortion perpendicular to the parabolas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The result of the remapping process, which includes the transformed image based on the specified parabolas and adjustments. This output is crucial for understanding the effect of the remapping on the original image.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapInsideParabolasAdvanced(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
            "curve_wise_adjust": ("FLOAT", {"default": 1, "min": .3, "max": 2, "step": .01}),
            "ortho_wise_adjust": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .01}),
            "flip_ortho": ("BOOLEAN", {"default": False})
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas, curve_wise_adjust, ortho_wise_adjust, flip_ortho):
        from .utils.remaps import remap_inside_parabolas_advanced
        return ({
                    "func": remap_inside_parabolas_advanced,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1),
                              curve_wise_adjust, ortho_wise_adjust, flip_ortho],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)

```
