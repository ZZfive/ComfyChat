# Multival Dynamic 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalDynamic`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

The ADE_MultivalDynamic node dynamically generates multivalued outputs based on specified float ranges and masks, with optional scaling. It's designed to facilitate the creation of varied outputs by applying linear or normalized scaling to masks, allowing for flexible manipulation of values within the Animate Diff framework.
## Input types
### Required
- **`float_val`**
    - Specifies the float value or a list of float values for dynamic range manipulation. It sets the base for the output values, enabling the generation of multivalued outputs.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, list[float]]`
### Optional
- **`mask_optional`**
    - An optional tensor representing the mask to be applied. The mask is used to selectively apply the scaling transformation to specific areas, enabling targeted manipulation of values.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`multival`**
    - Comfy dtype: `MULTIVAL`
    - The output of the node, which is a dynamically generated multivalued result based on the input parameters. It represents the conditioned values after applying the specified scaling to the mask.
    - Python dtype: `MultivalDynamicNode`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ADE_ApplyAnimateDiffModel](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_ApplyAnimateDiffModel.md)



## Source code
```python
class MultivalDynamicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001},),
            },
            "optional": {
                "mask_optional": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=1.0, mask_optional: Tensor=None):
        # first, normalize inputs
        # if float_val is iterable, treat as a list and assume inputs are floats
        float_is_iterable = False
        if isinstance(float_val, Iterable):
            float_is_iterable = True
            float_val = list(float_val)
            # if mask present, make sure float_val list can be applied to list - match lengths
            if mask_optional is not None:
                if len(float_val) < mask_optional.shape[0]:
                    # copies last entry enough times to match mask shape
                    float_val = float_val + float_val[-1]*(mask_optional.shape[0]-len(float_val))
                float_val = float_val[:mask_optional.shape[0]]
        # now that inputs are normalized, figure out what value to actually return
        if mask_optional is not None:
            mask_optional = mask_optional.clone()
            mask_optional = mask_optional * float_val
            return (mask_optional,)
        else:
            if not float_is_iterable:
                return (float_val,)
            # create a dummy mask of b,h,w=float_len,1,1 (sigle pixel)
            # purpose is for float input to work with mask code, without special cases
            float_len = len(float_val) if float_is_iterable else 1
            shape = (float_len,1,1)
            mask_optional = torch.ones(shape)
            mask_optional = mask_optional * float_val
            return (mask_optional,)

```
