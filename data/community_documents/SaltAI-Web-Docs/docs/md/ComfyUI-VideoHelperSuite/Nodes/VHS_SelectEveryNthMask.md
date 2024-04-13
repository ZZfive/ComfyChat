# Select Every Nth Mask 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthMask`
- Category: `Video Helper Suite 🎥🅥🅗🅢/mask`
- Output node: `False`

This node is designed to streamline the process of selecting every Nth mask from a batch of masks, facilitating operations that require periodic sampling or thinning of data for efficiency or specific analytical purposes.
## Input types
### Required
- **`mask`**
    - The input mask tensor from which every Nth mask will be selected. This parameter is crucial for defining the subset of masks to be processed.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`select_every_nth`**
    - Determines the interval at which masks are selected from the input batch, playing a key role in controlling the density of the output mask set.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`MASK`**
    - Comfy dtype: `MASK`
    - The output tensor containing the selected masks, providing a thinned or periodically sampled subset of the original mask batch.
    - Python dtype: `Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of masks selected and returned by the node, offering insight into the size of the processed subset.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "select_masks"

    def select_masks(self, mask: Tensor, select_every_nth: int):
        sub_mask = mask[0::select_every_nth]
        return (sub_mask, sub_mask.size(0))

```
