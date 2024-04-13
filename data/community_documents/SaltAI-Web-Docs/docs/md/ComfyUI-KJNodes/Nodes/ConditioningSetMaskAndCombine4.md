# ConditioningSetMaskAndCombine4
## Documentation
- Class name: `ConditioningSetMaskAndCombine4`
- Category: `KJNodes/masking/conditioning`
- Output node: `False`

This node is designed to apply masks to a set of conditioning data and combine them, allowing for the manipulation of the conditioning based on mask parameters and strengths. It supports multiple masks and conditioning sets, enabling complex conditioning scenarios.
## Input types
### Required
- **`positive_1`**
    - Represents the first set of positive conditioning data to which the mask is applied, influencing the conditioning's effect.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_1`**
    - Represents the first set of negative conditioning data, which is also subject to mask application, affecting its influence in the conditioning process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_2`**
    - The second set of positive conditioning data, which is modified by a corresponding mask to adjust its impact.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_2`**
    - The second set of negative conditioning data, which is modified by a corresponding mask to adjust its impact.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_3`**
    - The third set of positive conditioning data, further expanding the conditioning scenarios by applying another mask.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_3`**
    - The third set of negative conditioning data, also modified by a mask to influence the overall conditioning.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`positive_4`**
    - Represents the fourth set of positive conditioning data, allowing for even more complex conditioning scenarios through mask application.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`negative_4`**
    - The fourth set of negative conditioning data, which is influenced by mask application, contributing to the conditioning's complexity.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`mask_1`**
    - The first mask applied to the conditioning data, determining the area and intensity of the conditioning's effect.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_2`**
    - The second mask applied to the conditioning data, determining the area and intensity of the conditioning's effect.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_3`**
    - The third mask, further defining the conditioning's area and intensity of effect.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_4`**
    - The fourth mask, adding another layer of specificity to the conditioning's area and intensity.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_1_strength`**
    - Determines the intensity of the first mask's effect on the conditioning data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_2_strength`**
    - Determines the intensity of the second mask's effect on the conditioning data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_3_strength`**
    - Sets the intensity of the third mask's effect, further customizing the conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_4_strength`**
    - Adjusts the intensity of the fourth mask's effect, allowing for detailed conditioning customization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Specifies how the mask is applied to the conditioning data, either using default settings or mask bounds.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`combined_positive`**
    - Comfy dtype: `CONDITIONING`
    - The combined set of positive conditioning data after mask application and adjustments.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`combined_negative`**
    - Comfy dtype: `CONDITIONING`
    - The combined set of negative conditioning data after mask application and adjustments.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetMaskAndCombine4:
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
                "positive_4": ("CONDITIONING", ),
                "negative_4": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"

    def append(self, positive_1, negative_1, positive_2, positive_3, positive_4, negative_2, negative_3, negative_4, mask_1, mask_2, mask_3, mask_4, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength):
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
        if len(mask_4.shape) < 3:
            mask_4 = mask_4.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in positive_4:
            append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        for t in negative_4:
            append_helper(t, mask_4, c2, set_area_to_bounds, mask_4_strength)
        return (c, c2)

```
