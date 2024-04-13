# ConditioningSetMaskAndCombine
## Documentation
- Class name: `ConditioningSetMaskAndCombine`
- Category: `KJNodes/masking/conditioning`
- Output node: `False`

This node is designed to apply masks to conditioning data and combine them, adjusting the strength of the mask application. It enables the customization of conditioning areas within the data, providing a flexible approach to data manipulation and enhancement.
## Input types
### Required
- **`positive_1`**
    - The first set of positive conditioning data to which masks will be applied. It's part of the data that will be enhanced or modified.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_1`**
    - The first set of negative conditioning data to which masks will be applied. It represents the data that might be suppressed or altered negatively.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_2`**
    - The second set of positive conditioning data to which masks will be applied, contributing to the data enhancement.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_2`**
    - The second set of negative conditioning data to which masks will be applied, contributing to the data alteration.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`mask_1`**
    - The first mask to be applied to the conditioning data, defining specific areas within the data that will be affected.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_2`**
    - The second mask to be applied to the conditioning data, further defining areas to be affected.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_1_strength`**
    - Determines the intensity of the first mask's application on the conditioning data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_2_strength`**
    - Determines the intensity of the second mask's application on the conditioning data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Specifies whether to set the conditioning area to the bounds defined by the mask or use a default setting.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`combined_positive`**
    - Comfy dtype: `CONDITIONING`
    - A list of combined positive conditioning data with applied masks.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`combined_negative`**
    - Comfy dtype: `CONDITIONING`
    - A list of combined negative conditioning data with applied masks.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetMaskAndCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"

    def append(self, positive_1, negative_1, positive_2, negative_2, mask_1, mask_2, set_cond_area, mask_1_strength, mask_2_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        return (c, c2)

```
