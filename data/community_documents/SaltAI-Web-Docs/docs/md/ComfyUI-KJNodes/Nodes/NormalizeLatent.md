# NormalizeLatent
## Documentation
- Class name: `NormalizeLatent`
- Category: `KJNodes/noise`
- Output node: `True`

The NormalizeLatent node standardizes the distribution of latent samples by adjusting their scale to have a standard deviation of 1. This normalization process is crucial for stabilizing the behavior of subsequent operations on the latent samples.
## Input types
### Required
- **`latent`**
    - The latent input represents the latent samples to be normalized. It plays a crucial role in ensuring that the latent space is uniformly scaled, which is essential for consistent manipulation and generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a normalized version of the input latent samples, with their scale adjusted to have a standard deviation of 1.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NormalizeLatent:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "normalize"
    CATEGORY = "KJNodes/noise"
    OUTPUT_NODE = True

    def normalize(self, latent):
        samples = latent["samples"]
        samples /= samples.std()
        out = latent.copy()
        out["samples"] = samples
        return (out,)

```
