# 🔧 Mask Blur
## Documentation
- Class name: `MaskBlur+`
- Category: `essentials`
- Output node: `False`

The MaskBlur node applies a Gaussian blur to a given mask, allowing for the softening of its edges. This operation can be useful in creating smoother transitions and effects within images.
## Input types
### Required
- **`mask`**
    - The mask to be blurred. This input is crucial for determining the areas of the image that will undergo the blurring effect.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - Specifies the intensity of the blur effect. A higher value results in a more pronounced blurring.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resulting mask after applying the Gaussian blur. This output can be used for further image processing tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("FLOAT", { "default": 6.0, "min": 0, "step": 0.5, }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, amount):
        size = int(6 * amount +1)
        if size % 2 == 0:
            size+= 1
        
        blurred = mask.unsqueeze(1)
        blurred = T.GaussianBlur(size, amount)(blurred)
        blurred = blurred.squeeze(1)

        return(blurred,)

```
