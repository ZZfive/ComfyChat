# ConditioningSetMaskAndCombine5
## Documentation
- Class name: `ConditioningSetMaskAndCombine5`
- Category: `KJNodes/masking/conditioning`
- Output node: `False`

This node is designed to apply masks to conditioning data, modify the conditioning based on the masks, and then combine the modified conditioning data. It supports multiple sets of conditioning and masks, allowing for complex conditioning scenarios. The node emphasizes the ability to adjust the influence of each mask through strength parameters and to specify whether the conditioning should be set based on the mask bounds.
## Input types
### Required
- **`positive_1`**
    - Represents the first set of conditioning data to be modified and combined. It affects the node's execution by providing the initial data that will be masked and adjusted.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`negative_1`**
    - Represents the first set of conditioning data to be negatively modified and combined. It affects the node's execution by providing additional data that will be masked and adjusted in a manner opposite to the positive sets.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`positive_2`**
    - Represents the second set of conditioning data to be modified and combined. It provides additional data for masking and adjustment, contributing to the complexity of the conditioning scenario.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`negative_2`**
    - Represents the second set of conditioning data to be negatively modified and combined. It provides additional data for negative masking and adjustment, enhancing the diversity of the conditioning scenario.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`positive_3`**
    - Represents the third set of conditioning data to be modified and combined. It adds further complexity to the conditioning scenario by introducing more data for masking and adjustment.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`negative_3`**
    - Represents the third set of conditioning data to be negatively modified and combined. It further diversifies the conditioning scenario by adding more data for negative masking and adjustment.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`positive_4`**
    - Represents the fourth set of conditioning data to be modified and combined. It introduces additional complexity to the conditioning scenario.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`negative_4`**
    - Represents the fourth set of conditioning data to be negatively modified and combined. It adds more diversity to the conditioning scenario by providing another layer of data for negative masking and adjustment.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`positive_5`**
    - Represents the fifth set of conditioning data to be modified and combined. It extends the complexity of the conditioning scenario by adding more data for masking and adjustment.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`negative_5`**
    - Represents the fifth set of conditioning data to be negatively modified and combined. It further diversifies the conditioning scenario by introducing additional data for negative masking and adjustment.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`mask_1`**
    - The first mask to be applied to the conditioning data. It plays a crucial role in defining the areas of the conditioning data that will be modified.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_2`**
    - The second mask to be applied to the conditioning data. It allows for additional areas of the conditioning data to be modified, contributing to the node's flexibility.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_3`**
    - The third mask to be applied to the conditioning data. It enables further modification of the conditioning data, increasing the node's adaptability to complex scenarios.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_4`**
    - The fourth mask to be applied to the conditioning data. It allows for further areas of the conditioning data to be modified, enhancing the node's adaptability.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_5`**
    - The fifth mask to be applied to the conditioning data. It enables additional modification of the conditioning data, increasing the node's flexibility in handling complex scenarios.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask_1_strength`**
    - Adjusts the influence of the first mask on the conditioning data. It allows for fine-tuning the impact of the mask, enabling precise control over the conditioning modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_2_strength`**
    - Adjusts the influence of the second mask on the conditioning data. It provides additional control over how the conditioning is modified, allowing for nuanced adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_3_strength`**
    - Adjusts the influence of the third mask on the conditioning data. It offers further customization of the conditioning modification, facilitating detailed control.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_4_strength`**
    - Adjusts the influence of the fourth mask on the conditioning data. It provides further control over the conditioning modification, allowing for detailed adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_5_strength`**
    - Adjusts the influence of the fifth mask on the conditioning data. It offers additional customization of the conditioning modification, facilitating precise control.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Determines whether the conditioning should be set based on the mask bounds. This parameter influences how the masks are applied and the extent of their effect on the conditioning data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`combined_positive`**
    - Comfy dtype: `CONDITIONING`
    - The combined and modified positive conditioning data after applying the masks and adjustments.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`combined_negative`**
    - Comfy dtype: `CONDITIONING`
    - The combined and modified negative conditioning data after applying the masks and adjustments.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetMaskAndCombine5:
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
                "positive_5": ("CONDITIONING", ),
                "negative_5": ("CONDITIONING", ),
                "mask_1": ("MASK", ),
                "mask_2": ("MASK", ),
                "mask_3": ("MASK", ),
                "mask_4": ("MASK", ),
                "mask_5": ("MASK", ),
                "mask_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mask_5_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("combined_positive", "combined_negative",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/masking/conditioning"

    def append(self, positive_1, negative_1, positive_2, positive_3, positive_4, positive_5, negative_2, negative_3, negative_4, negative_5, mask_1, mask_2, mask_3, mask_4, mask_5, set_cond_area, mask_1_strength, mask_2_strength, mask_3_strength, mask_4_strength, mask_5_strength):
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
        if len(mask_5.shape) < 3:
            mask_5 = mask_5.unsqueeze(0)
        for t in positive_1:
            append_helper(t, mask_1, c, set_area_to_bounds, mask_1_strength)
        for t in positive_2:
            append_helper(t, mask_2, c, set_area_to_bounds, mask_2_strength)
        for t in positive_3:
            append_helper(t, mask_3, c, set_area_to_bounds, mask_3_strength)
        for t in positive_4:
            append_helper(t, mask_4, c, set_area_to_bounds, mask_4_strength)
        for t in positive_5:
            append_helper(t, mask_5, c, set_area_to_bounds, mask_5_strength)
        for t in negative_1:
            append_helper(t, mask_1, c2, set_area_to_bounds, mask_1_strength)
        for t in negative_2:
            append_helper(t, mask_2, c2, set_area_to_bounds, mask_2_strength)
        for t in negative_3:
            append_helper(t, mask_3, c2, set_area_to_bounds, mask_3_strength)
        for t in negative_4:
            append_helper(t, mask_4, c2, set_area_to_bounds, mask_4_strength)
        for t in negative_5:
            append_helper(t, mask_5, c2, set_area_to_bounds, mask_5_strength)
        return (c, c2)

```
