# 🔧 Mask From Batch
## Documentation
- Class name: `MaskFromBatch+`
- Category: `essentials`
- Output node: `False`

This node is designed to extract a specific segment from a given mask based on a starting point and a specified length. It aims to facilitate operations on subsets of masks, enhancing flexibility in processing and manipulation of mask data.
## Input types
### Required
- **`mask`**
    - The mask from which a segment will be extracted. This parameter is crucial for defining the subset of the mask to be operated on.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`start`**
    - Specifies the starting index from which the mask segment should be extracted. It determines the initial point of the segment within the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`length`**
    - Defines the length of the mask segment to be extracted. If set to a negative value, it indicates that the segment should extend to the end of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The extracted segment of the mask. This output facilitates further operations or analysis on the specified mask segment.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, start, length):
        if length<0:
            length = mask.shape[0]
        start = min(start, mask.shape[0]-1)
        length = min(mask.shape[0]-start, length)
        return (mask[start:start + length], )

```
