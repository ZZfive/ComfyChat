# 🔧 Image Contrast Adaptive Sharpening
## Documentation
- Class name: `ImageCASharpening+`
- Category: `essentials`
- Output node: `False`

This node applies contrast adaptive sharpening to an image, enhancing its details and improving visual clarity without significantly altering the overall composition.
## Input types
### Required
- **`image`**
    - The input image to be sharpened. It's crucial for defining the base upon which the sharpening effect will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - Specifies the intensity of the sharpening effect. A higher value results in a more pronounced sharpening effect, allowing for finer control over the image's final appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is the sharpened version of the input image, with enhanced details and improved clarity.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageScaleBy](../../Comfy/Nodes/ImageScaleBy.md)
    - [ImpactSimpleDetectorSEGS_for_AD](../../ComfyUI-Impact-Pack/Nodes/ImpactSimpleDetectorSEGS_for_AD.md)



## Source code
```python
class ImageCAS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials"
    FUNCTION = "execute"

    def execute(self, image, amount):
        img = F.pad(p(image), pad=(1, 1, 1, 1)).cpu()

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]
        
        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)
        
        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2
        
        # Computing local weight
        inv_mx = torch.reciprocal(mx + EPSILON)
        amp = inv_mx * torch.minimum(mn, (2 - mx))
    
        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)   # this seems the only way to ensure there are no NaNs

        output = pb(output) 

        return (output,)

```
