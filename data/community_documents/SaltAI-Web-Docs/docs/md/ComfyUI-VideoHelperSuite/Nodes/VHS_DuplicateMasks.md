# Duplicate Mask Batch 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_DuplicateMasks`
- Category: `Video Helper Suite 🎥🅥🅗🅢/mask`
- Output node: `False`

The VHS_DuplicateMasks node is designed to replicate a given mask tensor a specified number of times, effectively creating a batch of identical masks. This functionality is useful in scenarios where multiple instances of the same mask are required for batch processing or augmentation purposes.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input mask tensor that is to be duplicated. It is central to the node's operation, as it defines the mask that will be replicated.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`multiply_by`**
    - The 'multiply_by' parameter specifies the number of times the input mask should be duplicated. It directly influences the size of the output batch of masks, enabling dynamic adjustment based on processing needs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`MASK`**
    - Comfy dtype: `MASK`
    - The output is a tensor containing the duplicated masks, arranged in a batch. This facilitates further processing or analysis of the masks in a batched manner.
    - Python dtype: `torch.Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - This output represents the total number of masks in the output batch, providing a straightforward way to ascertain the batch size.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DuplicateMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, mask: Tensor, multiply_by: int):
        full_masks = []
        for n in range(0, multiply_by):
            full_masks.append(mask)
        new_mask = torch.cat(full_masks, dim=0)
        return (new_mask, new_mask.size(0),)

```
