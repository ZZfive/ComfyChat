# 🔧 Mask Expand Batch
## Documentation
- Class name: `MaskExpandBatch+`
- Category: `essentials`
- Output node: `False`

The node is designed to manipulate the size of masks in a batch by either expanding or contracting them, optionally with tapered corners for a smoother transition. This functionality is crucial for adjusting mask dimensions to fit specific requirements or to achieve desired effects in image processing tasks.
## Input types
### Required
- **`mask`**
    - The mask input represents the mask or masks to be manipulated. It is central to the node's operation as it determines the base data on which size adjustments are performed.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`size`**
    - The size parameter specifies the target size of the mask batch after processing. It allows for the adjustment of the batch size to meet specific requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`method`**
    - The method parameter determines the approach used to adjust the size of the mask batch, including options like 'expand', 'all', 'first', and 'last', each offering different ways of resizing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask or a batch of masks that have been resized according to the specified parameters, maintaining the original content while adjusting dimensions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, size, method):
        orig_size = mask.shape[0]

        if orig_size == size:
            return (mask,)

        if size <= 1:
            return (mask[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(mask.shape)[1:], dtype=mask.dtype, device=mask.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = mask[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = mask[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = mask.repeat([math.ceil(size / mask.shape[0])] + [1] * (len(mask.shape) - 1))[:size]
        elif 'first' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat([mask[:1].repeat(size-mask.shape[0], 1, 1), mask], dim=0)
        elif 'last' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat((mask, mask[-1:].repeat((size-mask.shape[0], 1, 1))), dim=0)

        return (out,)

```
