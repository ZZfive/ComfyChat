# Blur (mtb)
## Documentation
- Class name: `Blur (mtb)`
- Category: `mtb/image processing`
- Output node: `False`

This node applies a Gaussian blur to an image, utilizing separate horizontal and vertical sigma values to control the blur intensity and direction.
## Input types
### Required
- **`image`**
    - The image to be blurred. This parameter is crucial as it directly influences the visual outcome of the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`sigmaX`**
    - Controls the blur intensity along the X-axis, affecting the horizontal spread of the blur.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigmaY`**
    - Controls the blur intensity along the Y-axis, affecting the vertical spread of the blur.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The blurred image, which is the result of applying a Gaussian filter with the specified sigma values.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [Mix Images By Mask](../../masquerade-nodes-comfyui/Nodes/Mix Images By Mask.md)



## Source code
```python
class Blur_:
    """Blur an image using a Gaussian filter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigmaX": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "sigmaY": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"
    CATEGORY = "mtb/image processing"

    def blur(self, image: torch.Tensor, sigmaX, sigmaY):
        image = image.numpy()
        image = image.transpose(1, 2, 3, 0)
        image = gaussian(image, sigma=(sigmaX, sigmaY, 0, 0))
        image = image.transpose(3, 0, 1, 2)
        return (torch.from_numpy(image),)

```
