# RemapBarrelDistortion
## Documentation
- Class name: `RemapBarrelDistortion`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapBarrelDistortion node applies a barrel distortion effect to images, which can be used for correcting lens distortion or creating artistic effects. It allows for precise control over the distortion through parameters that define the curvature of the effect.
## Input types
### Required
- **`a`**
    - Defines the coefficient for the cubic term in the distortion equation, influencing the curvature of the barrel distortion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b`**
    - Defines the coefficient for the quadratic term in the distortion equation, affecting the shape of the barrel distortion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`c`**
    - Defines the coefficient for the linear term in the distortion equation, adjusting the intensity of the distortion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`d`**
    - Optional parameter that defines the constant term in the distortion equation, allowing for additional fine-tuning of the distortion effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - Produces the coordinates for remapping the image to achieve the barrel distortion effect. This output is essential for visualizing or further processing the distorted image.
    - Python dtype: `Tuple[ndarray, ndarray, NoneType]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapBarrelDistortion(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "a": ("FLOAT", {"default": 0, "step": 0.00001}),
                "b": ("FLOAT", {"default": 0, "step": 0.00001}),
                "c": ("FLOAT", {"default": 0, "step": 0.00001}),
            },
            "optional": {
                "d": ("FLOAT", {"forceInput": True})
            }
        }

    def send_remap(self, a, b, c, d=None):
        from .utils.remaps import remap_barrel_distortion
        return ({
                    "func": remap_barrel_distortion,
                    "xargs": [a, b, c, d]
                },)

```
