# Conditioning (Set Area with Percentage)
## Documentation
- Class name: `ConditioningSetAreaPercentage`
- Category: `conditioning`
- Output node: `False`

The ConditioningSetAreaPercentage node specializes in adjusting the area of influence for conditioning elements based on percentage values. It allows for the specification of the area's dimensions and position as percentages of the total image size, alongside a strength parameter to modulate the intensity of the conditioning effect.
## Input types
### Required
- **`conditioning`**
    - Represents the conditioning elements to be modified, serving as the foundation for applying area and strength adjustments.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`width`**
    - Specifies the width of the area as a percentage of the total image width, influencing how much of the image the conditioning affects horizontally.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`height`**
    - Determines the height of the area as a percentage of the total image height, affecting the vertical extent of the conditioning's influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`x`**
    - Indicates the horizontal starting point of the area as a percentage of the total image width, positioning the conditioning effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`y`**
    - Specifies the vertical starting point of the area as a percentage of the total image height, positioning the conditioning effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength`**
    - Controls the intensity of the conditioning effect within the specified area, allowing for fine-tuning of its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Returns the modified conditioning elements with updated area and strength parameters, ready for further processing or application.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetAreaPercentage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "height": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                              "x": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "y": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['area'] = ("percentage", height, width, y, x)
            n[1]['strength'] = strength
            n[1]['set_area_to_bounds'] = False
            c.append(n)
        return (c, )

```
