# Latent Lerp (mtb)
## Documentation
- Class name: `Latent Lerp (mtb)`
- Category: `mtb/latent`
- Output node: `False`

The LatentLerp node performs linear interpolation between two latent vectors, blending them together based on a specified ratio. This operation is useful for generating intermediate representations between two given states in a latent space.
## Input types
### Required
- **`A`**
    - The first latent vector to be interpolated. It serves as the starting point for the linear interpolation.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`B`**
    - The second latent vector to be interpolated. It serves as the endpoint for the linear interpolation.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`t`**
    - The interpolation ratio, determining the blend between the first and second latent vectors. A value of 0 corresponds to the first vector, while 1 corresponds to the second.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The interpolated latent vector, representing a blend between the input latent vectors based on the specified ratio.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LatentLerp:
    """Linear interpolation (blend) between two latent vectors"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("LATENT",),
                "B": ("LATENT",),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "lerp_latent"

    CATEGORY = "mtb/latent"

    def lerp_latent(self, A, B, t):
        a = A.copy()
        b = B.copy()

        torch.lerp(a["samples"], b["samples"], t, out=a["samples"])

        return (a,)

```
