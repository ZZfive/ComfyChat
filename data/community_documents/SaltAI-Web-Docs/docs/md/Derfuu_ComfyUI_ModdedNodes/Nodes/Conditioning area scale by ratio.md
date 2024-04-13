# Conditioning area scale by ratio
## Documentation
- Class name: `Conditioning area scale by ratio`
- Category: `Derfuu_Nodes/Modded nodes/Conditions`
- Output node: `False`

This node is designed to adjust the scale of conditioning areas by a specified ratio, modifying both the size and strength of the conditioning to achieve a desired effect. It allows for the dynamic resizing of conditioning areas, making it a versatile tool for fine-tuning the influence of conditioning on the generation process.
## Input types
### Required
- **`conditioning`**
    - The conditioning input represents the current state of conditioning areas that will be scaled. It is crucial for determining the base from which modifications will be applied.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`modifier`**
    - The modifier is a scaling factor that adjusts the size of the conditioning areas. It plays a key role in determining the extent of the resizing effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_modifier`**
    - This parameter adjusts the strength of the conditioning, allowing for finer control over the impact of the conditioning areas after resizing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Returns the modified conditioning areas with adjusted sizes and strengths, reflecting the applied scaling ratio.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningAreaScale_Ratio:
    def __init__(self):
        pass


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ('CONDITIONING',),
                "modifier": field.FLOAT,
                "strength_modifier": field.FLOAT,
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "resize"
    CATEGORY = TREE_COND

    def resize(self, conditioning, modifier, strength_modifier, min_sigma=0.0, max_sigma=99.0):
        c = []

        for t in conditioning:
            n = [t[0], t[1].copy()]

            try:
                size, offset = sizes.get_conditioning_size(n[1])
            except:
                c.append(n)
                continue

            height = int(size["height"] * 8 * modifier)
            width = int(size["width"] * 8 * modifier)

            y = int(offset["y_offset"] * 8 * modifier)
            x = int(offset["x_offset"] * 8 * modifier)

            n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            n[1]['strength'] *= strength_modifier
            n[1]['min_sigma'] = min_sigma
            n[1]['max_sigma'] = max_sigma
            c.append(n)

        return (c,)

```
