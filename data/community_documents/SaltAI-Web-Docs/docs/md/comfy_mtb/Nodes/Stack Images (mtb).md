# Stack Images (mtb)
## Documentation
- Class name: `Stack Images (mtb)`
- Category: `mtb/image utils`
- Output node: `False`

The Stack Images node is designed to combine multiple input images into a single image, either by stacking them horizontally or vertically, based on the specified orientation. This functionality is essential for creating composite images or for visualization purposes where multiple images need to be compared side by side or in a sequence.
## Input types
### Required
- **`vertical`**
    - Determines the orientation of the stacking process. If set to True, images are stacked vertically; otherwise, they are stacked horizontally. This affects the final layout of the combined image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single tensor representing the stacked image, which combines all input images according to the specified orientation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class StackImages:
    """Stack the input images horizontally or vertically"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vertical": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stack"
    CATEGORY = "mtb/image utils"

    def stack(self, vertical, **kwargs):
        if not kwargs:
            raise ValueError("At least one tensor must be provided.")

        tensors = list(kwargs.values())
        log.debug(
            f"Stacking {len(tensors)} tensors {'vertically' if vertical else 'horizontally'}"
        )
        log.debug(list(kwargs.keys()))

        ref_shape = tensors[0].shape
        for tensor in tensors[1:]:
            if tensor.shape[1:] != ref_shape[1:]:
                raise ValueError(
                    "All tensors must have the same dimensions except for the stacking dimension."
                )

        dim = 1 if vertical else 2

        stacked_tensor = torch.cat(tensors, dim=dim)

        return (stacked_tensor,)

```
