# Split Mask Batch 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SplitMasks`
- Category: `Video Helper Suite 🎥🅥🅗🅢/mask`
- Output node: `False`

The VHS_SplitMasks node is designed for dividing a batch of masks into two groups based on a specified index. This functionality is essential for workflows that require the separation of mask data into distinct sets for further processing or analysis.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the batch of masks to be split. It is crucial for determining how the masks are divided into two groups.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`split_index`**
    - The 'split_index' parameter specifies the index at which the batch of masks is split into two groups. Its value influences the distribution of masks between the two resulting sets.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`MASK_A`**
    - Comfy dtype: `MASK`
    - The first group of masks obtained after the split.
    - Python dtype: `Tensor`
- **`A_count`**
    - Comfy dtype: `INT`
    - The count of masks in the first group.
    - Python dtype: `int`
- **`MASK_B`**
    - Comfy dtype: `MASK`
    - The second group of masks obtained after the split.
    - Python dtype: `Tensor`
- **`B_count`**
    - Comfy dtype: `INT`
    - The count of masks in the second group.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SplitMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": -99999999999}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT", "MASK", "INT")
    RETURN_NAMES = ("MASK_A", "A_count", "MASK_B", "B_count")
    FUNCTION = "split_masks"

    def split_masks(self, mask: Tensor, split_index: int):
        group_a = mask[:split_index]
        group_b = mask[split_index:]
        return (group_a, group_a.size(0), group_b, group_b.size(0))

```
