# Multival Scaled Mask 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalScaledMask`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

This node is designed to dynamically adjust the scale of masks based on specified minimum and maximum float values, potentially incorporating additional scaling strategies. It aims to facilitate the creation of multivalued outputs by applying these scales to masks, thereby enabling more nuanced and flexible manipulation of mask properties within the context of animation and differential effects.
## Input types
### Required
- **`min_float_val`**
    - Specifies the minimum float value for scaling the mask. It sets the lower bound for the scale adjustment, ensuring that the mask's scale does not fall below this value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_float_val`**
    - Defines the maximum float value for scaling the mask. This value sets the upper limit for the scale adjustment, ensuring that the mask's scale does not exceed this threshold.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask`**
    - The mask to be scaled. This input is crucial for determining the base mask that will undergo scaling adjustments based on the specified minimum and maximum float values.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
### Optional
- **`scaling`**
    - An optional parameter that specifies the scaling strategy to be applied. It can influence how the mask is scaled between the minimum and maximum float values, offering additional control over the scaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`multival`**
    - Comfy dtype: `MULTIVAL`
    - The output is a multivalued version of the input mask, adjusted according to the specified scaling parameters and strategy. It represents the mask after being scaled between the minimum and maximum float values.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MultivalScaledMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_float_val": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001}),
                "max_float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "mask": ("MASK",),
            },
            "optional": {
                "scaling": (ScaleType.LIST,),
            }
        }

    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, min_float_val: float, max_float_val: float, mask: Tensor, scaling: str=ScaleType.ABSOLUTE):
        if isinstance(min_float_val, Iterable):
            raise ValueError(f"min_float_val must be type float (no lists allowed here), not {type(min_float_val).__name__}.")
        if isinstance(max_float_val, Iterable):
            raise ValueError(f"max_float_val must be type float (no lists allowed here), not {type(max_float_val).__name__}.")
        
        if scaling == ScaleType.ABSOLUTE:
            mask = linear_conversion(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        elif scaling == ScaleType.RELATIVE:
            mask = normalize_min_max(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        else:
            raise ValueError(f"scaling '{scaling}' not recognized.")
        return MultivalDynamicNode.create_multival(self, mask_optional=mask)

```
