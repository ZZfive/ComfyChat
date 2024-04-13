# Image Mix
## Documentation
- Class name: `JWImageMix`
- Category: `jamesWalker55`
- Output node: `False`

The JWImageMix node blends two images together based on a specified blend type and factor, allowing for creative image manipulation and combination.
## Input types
### Required
- **`blend_type`**
    - Specifies the method of blending the two images. Can be either 'mix' for linear interpolation or 'multiply' for a multiplication blend, affecting the visual outcome of the blend.
    - Comfy dtype: `['mix', 'multiply']`
    - Python dtype: `str`
- **`factor`**
    - Determines the weight of the blend between the two images, influencing the dominance of one image over the other in the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_a`**
    - The first image to be blended. Serves as the base image for the blending operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_b`**
    - The second image to be blended with the first. Its influence is determined by the blend type and factor.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of blending the two input images according to the specified blend type and factor.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
@register_node("JWImageMix", "Image Mix")
class _:
    CATEGORY = "jamesWalker55"
    BLEND_TYPES = ("mix", "multiply")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_type": (cls.BLEND_TYPES, {"default": "mix"}),
                "factor": ("FLOAT", {"min": 0, "max": 1, "step": 0.01, "default": 0.5}),
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(
        self,
        blend_type: str,
        factor: float,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ):
        assert blend_type in self.BLEND_TYPES
        assert isinstance(factor, float)
        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        assert image_a.shape == image_b.shape

        if blend_type == "mix":
            mixed = image_a * (1 - factor) + image_b * factor
        elif blend_type == "multiply":
            mixed = image_a * (1 - factor + image_b * factor)
        else:
            raise NotImplementedError(f"Blend type not yet implemented: {blend_type}")

        return (mixed,)

```
