# Bitwise(MASK - MASK)
## Documentation
- Class name: `SubtractMask`
- Category: `ImpactPack/Operation`
- Output node: `False`

The SubtractMask node is designed to perform a subtraction operation between two given masks. It abstracts the complexity of handling mask data types and operations, providing a straightforward way to obtain the difference between two masks, which can be crucial in image processing tasks where distinguishing between varying elements is necessary.
## Input types
### Required
- **`mask1`**
    - The first mask to be subtracted from. It plays a crucial role in determining the base of the subtraction operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask2`**
    - The second mask to subtract from the first one. This mask's values are subtracted from the first mask's values, influencing the final result of the operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The result of the subtraction operation between the two input masks, represented as a mask where each pixel value reflects the difference.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SubtractMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask1": ("MASK", ),
                        "mask2": ("MASK", ),
                      }
                }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask1, mask2):
        mask = subtract_masks(mask1, mask2)
        return (mask,)

```
