# ConditioningSetAreaStrength
## Documentation
- Class name: `ConditioningSetAreaStrength`
- Category: `conditioning`
- Output node: `False`

This node is designed to modify the strength attribute of a given conditioning set, allowing for the adjustment of the influence or intensity of the conditioning on the generation process.
## Input types
### Required
- **`conditioning`**
    - The conditioning set to be modified, representing the current state of conditioning that influences the generation process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`strength`**
    - The strength value to be applied to the conditioning set, dictating the intensity of its influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning set with updated strength values for each element.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetAreaStrength:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, strength):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['strength'] = strength
            c.append(n)
        return (c, )

```
