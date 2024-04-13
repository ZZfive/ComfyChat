# Conditioning (Set Mask)
## Documentation
- Class name: `ConditioningSetMask`
- Category: `conditioning`
- Output node: `False`

This node is designed to modify the conditioning of a generative model by applying a mask with a specified strength to certain areas. It allows for targeted adjustments within the conditioning, enabling more precise control over the generation process.
## Input types
### Required
- **`conditioning`**
    - The conditioning data to be modified. It serves as the basis for applying the mask and strength adjustments.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`mask`**
    - A mask tensor that specifies the areas within the conditioning to be modified.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - The strength of the mask's effect on the conditioning, allowing for fine-tuning of the applied modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Determines whether the mask's effect is applied to the default area or bounded by the mask itself, offering flexibility in targeting specific regions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning data, with the mask and strength adjustments applied.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - Attention couple
    - [ConditioningCombine](../../Comfy/Nodes/ConditioningCombine.md)



## Source code
```python
class ConditioningSetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                              "mask": ("MASK", ),
                              "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "set_cond_area": (["default", "mask bounds"],),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, conditioning, mask, set_cond_area, strength):
        c = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            _, h, w = mask.shape
            n[1]['mask'] = mask
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        return (c, )

```
