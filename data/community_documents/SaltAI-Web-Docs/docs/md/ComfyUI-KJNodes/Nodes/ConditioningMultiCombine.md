# Conditioning Multi Combine
## Documentation
- Class name: `ConditioningMultiCombine`
- Category: `KJNodes/masking/conditioning`
- Output node: `False`

The ConditioningMultiCombine node is designed to dynamically combine multiple conditioning inputs into a single conditioning output. It leverages a specified number of conditioning inputs and merges them sequentially, facilitating complex conditioning scenarios in generative models.
## Input types
### Required
- **`inputcount`**
    - Specifies the number of conditioning inputs to be combined. It determines the dynamic nature of the node, allowing for a flexible number of inputs to be processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`conditioning_1`**
    - The first conditioning input to be combined. It serves as the initial condition that subsequent conditions are merged with.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `CONDITIONING`
- **`conditioning_2`**
    - The second conditioning input to be combined. It is merged with the first conditioning input, starting the sequential combination process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `CONDITIONING`
## Output types
- **`combined`**
    - Comfy dtype: `CONDITIONING`
    - The combined conditioning output, resulting from the sequential merging of all specified conditioning inputs.
    - Python dtype: `CONDITIONING`
- **`inputcount`**
    - Comfy dtype: `INT`
    - The number of conditioning inputs that were combined. It reflects the dynamic input count specified by the user.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningMultiCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 20, "step": 1}),
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
            },
        
    }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("combined", "inputcount")
    FUNCTION = "combine"
    CATEGORY = "KJNodes/masking/conditioning"

    def combine(self, inputcount, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()
        cond = kwargs["conditioning_1"]
        for c in range(1, inputcount):
            new_cond = kwargs[f"conditioning_{c + 1}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]
        return (cond, inputcount,)

```
