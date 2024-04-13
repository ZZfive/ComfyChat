# RemapStretch
## Documentation
- Class name: `RemapStretch`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapStretch node applies a stretching transformation to an image based on specified power and center parameters. It utilizes a mathematical remapping technique to adjust the spatial distribution of pixels, effectively stretching the image in a controlled manner around a given center point.
## Input types
### Required
- **`power_x`**
    - Specifies the power of stretching along the x-axis. It determines the intensity of the stretch effect horizontally.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`power_y`**
    - Specifies the power of stretching along the y-axis. It determines the intensity of the stretch effect vertically.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_x`**
    - Defines the x-coordinate of the center around which the image will be stretched. This parameter sets the focal point of the stretching effect horizontally.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_y`**
    - Defines the y-coordinate of the center around which the image will be stretched. This parameter sets the focal point of the stretching effect vertically.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is a set of parameters or a function that defines how the image should be remapped or transformed, applying the stretching effect based on the input parameters.
    - Python dtype: `Dict[str, Any]`
- **`ui`**
    - The output is a transformed image with the applied stretching effect, based on the input parameters.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapStretch(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return RemapPinch.INPUT_TYPES_DICT

    def send_remap(self, power_x, power_y, center_x, center_y):
        from .utils.remaps import remap_pinch_or_stretch
        return ({
                    "func": remap_pinch_or_stretch,
                    "xargs": [(1/power_x, 1/power_y), (center_x, center_y)]
                },)

```
