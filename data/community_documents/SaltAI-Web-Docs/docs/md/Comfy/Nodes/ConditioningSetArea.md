# Conditioning (Set Area)
## Documentation
- Class name: `ConditioningSetArea`
- Category: `conditioning`
- Output node: `False`

This node is designed to modify the conditioning information by setting specific areas within the conditioning context. It allows for the precise spatial manipulation of conditioning elements, enabling targeted adjustments and enhancements based on specified dimensions and strength.
## Input types
### Required
- **`conditioning`**
    - The conditioning data to be modified. It serves as the base for applying spatial adjustments.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`width`**
    - Specifies the width of the area to be set within the conditioning context, influencing the horizontal scope of the adjustment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the area to be set, affecting the vertical extent of the conditioning modification.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`x`**
    - The horizontal starting point of the area to be set, positioning the adjustment within the conditioning context.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The vertical starting point for the area adjustment, establishing its position within the conditioning context.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Defines the intensity of the conditioning modification within the specified area, allowing for nuanced control over the adjustment's impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning data, reflecting the specified area settings and adjustments.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ConditioningCombine](../../Comfy/Nodes/ConditioningCombine.md)
    - SetNode



## Source code
```python
class ConditioningSetArea:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, width, height, x, y, strength):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['set_area_to_bounds'] = False
            c.append(n)
        return (c, )

```
