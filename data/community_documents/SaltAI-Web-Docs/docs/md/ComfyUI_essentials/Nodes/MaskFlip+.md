# 🔧 Mask Flip
## Documentation
- Class name: `MaskFlip+`
- Category: `essentials`
- Output node: `False`

The MaskFlip node is designed to flip a given mask along specified axes. It provides the functionality to manipulate the orientation of masks, which can be useful in various image processing and augmentation tasks.
## Input types
### Required
- **`mask`**
    - The mask to be flipped. This parameter is crucial as it defines the input mask that will undergo the flipping operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`axis`**
    - Specifies the axis or axes along which the mask will be flipped. It can be 'x', 'y', or 'xy', allowing for flexible manipulation of the mask's orientation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The flipped mask. This output is the result of flipping the input mask along the specified axes.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "axis": (["x", "y", "xy"],),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        mask = torch.flip(mask, dims=dim)

        return(mask,)

```
