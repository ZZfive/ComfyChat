# RoundMask
## Documentation
- Class name: `RoundMask`
- Category: `KJNodes/masking`
- Output node: `False`

The RoundMask node is designed to round the values of a given mask to the nearest integers, effectively normalizing its values for further processing or analysis.
## Input types
### Required
- **`mask`**
    - The mask to be rounded. This operation normalizes the mask's values to the nearest integers, which is crucial for ensuring consistency and accuracy in subsequent processing steps.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The rounded mask, with all values normalized to the nearest integers. This is essential for maintaining the integrity of the mask's structure for further operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RoundMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),  
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "round"
    CATEGORY = "KJNodes/masking"

    def round(self, mask):
        mask = mask.round()
        return (mask,)

```
