# Gaussian Blur Mask
## Documentation
- Class name: `ImpactGaussianBlurMask`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactGaussianBlurMask node applies a Gaussian blur to a given mask, utilizing specified kernel size and sigma values. This process smooths the edges of the mask, effectively blending it more seamlessly with its surroundings.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input mask to which the Gaussian blur will be applied. This is crucial for achieving a smoother, more integrated appearance of the mask within its context.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`kernel_size`**
    - The 'kernel_size' parameter determines the size of the Gaussian kernel used for blurring. A larger kernel size results in a more pronounced blurring effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma`**
    - The 'sigma' parameter controls the spread of the Gaussian blur. A higher sigma value increases the blurring effect, making the mask edges softer and more diffused.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a modified version of the input mask, now smoothed by the Gaussian blur. This processed mask blends more naturally with its surroundings.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GaussianBlurMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "mask": ("MASK", ),
                     "kernel_size": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                     "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                }}

    RETURN_TYPES = ("MASK", )

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, mask, kernel_size, sigma):
        mask = torch.unsqueeze(mask, dim=-1)
        mask = utils.tensor_gaussian_blur_mask(mask, kernel_size, sigma)
        mask = torch.squeeze(mask, dim=-1)
        return (mask, )

```
