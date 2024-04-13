# ConditioningSetMaskAndCombine3
## Documentation
- Class name: `ConditioningSetMaskAndCombine3`
- Category: `KJNodes/masking/conditioning`
- Output node: `False`

This node is designed to apply masks to conditioning data, combining them based on specified conditions and strengths. It enables the dynamic adjustment of conditioning areas through masks, facilitating the targeted manipulation of data for enhanced control and customization.
## Input types
### Required
- **`positive_1`**
    - Represents the first set of conditioning data to which the first mask is applied. It plays a crucial role in determining the initial state of the combined conditioning output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_1`**
    - Represents the first set of conditioning data to which the first mask is applied inversely, affecting the combined conditioning output by negating specific areas.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_2`**
    - Represents the second set of conditioning data to which the second mask is applied, further modifying the combined conditioning output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_2`**
    - Represents the second set of conditioning data to which the second mask is applied inversely, further negating specific areas in the combined conditioning output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_3`**
    - Represents the third set of conditioning data to which the third mask is applied, adding another layer of modification to the combined conditioning output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_3`**
    - Represents the third set of conditioning data to which the third mask is applied inversely, adding another layer of negation to the combined conditioning output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`mask_1`**
    - The first mask applied to the conditioning data, determining the areas of effect for the first set of positive and negative conditioning data.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_2`**
    - The second mask applied to the conditioning data, determining the areas of effect for the second set of positive and negative conditioning data.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_3`**
    - The third mask applied to the conditioning data, determining the areas of effect for the third set of positive and negative conditioning data.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_1_strength`**
    - Defines the strength of the first mask's effect on the conditioning data, allowing for fine-tuned control over the impact of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_2_strength`**
    - Defines the strength of the second mask's effect on the conditioning data, allowing for fine-tuned control over the impact of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_3_strength`**
    - Defines the strength of the third mask's effect on the conditioning data, allowing for fine-tuned control over the impact of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Determines whether the conditioning area is set to the bounds of the mask or uses a default setting, influencing how the masks are applied to the conditioning data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`combined_positive`**
    - Comfy dtype: `CONDITIONING`
    - The output is a list of positively conditioned data, each element being a tuple containing the original data and a dictionary of modifications applied based on the masks and their strengths.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`combined_negative`**
    - Comfy dtype: `CONDITIONING`
    - The output is a list of negatively conditioned data, each element being a tuple containing the original data and a dictionary of modifications applied inversely based on the masks and their strengths.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetMaskAndCombine3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_1": ("CONDITIONING", ),
                "negative_1": ("CONDITIONING", ),
                "positive_2": ("CONDITIONING", ),
                "negative_2": ("CONDITIONING", ),
                "positive_3": ("CONDITIONING", ),
                "negative_3": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"

    def append(self, positive_1, negative_1, positive_2, positive_3, negative_2, negative_3, mask_1, mask_2, mask_3, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength):
        c = []
        c2 = []
        set_area_to_bounds = False
        if set_cond_area != "default":
            set_area_to_bounds = True
        if len(mask_1.shape) < 3:
            mask_1 = mask_1.unsqueeze(0)
        if len(mask_2.shape) < 3:
            mask_2 = mask_2.unsqueeze(0)
        if len(mask_3.shape) < 3:
            mask_3 = mask_3.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        return (c, c2)

```
