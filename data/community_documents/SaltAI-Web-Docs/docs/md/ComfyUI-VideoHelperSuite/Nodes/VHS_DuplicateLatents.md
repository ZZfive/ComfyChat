# Duplicate Latent Batch 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_DuplicateLatents`
- Category: `Video Helper Suite 🎥🅥🅗🅢/latent`
- Output node: `False`

This node is designed to duplicate a batch of latent representations a specified number of times. It plays a crucial role in data augmentation or when a larger dataset is needed from a smaller set of latents for various processing or training purposes.
## Input types
### Required
- **`latents`**
    - The input latent representations to be duplicated. This parameter is essential for defining the starting point of the duplication process.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict[str, torch.Tensor]`
- **`multiply_by`**
    - Specifies the number of times the input latents should be duplicated. This parameter directly influences the size of the output dataset, allowing for flexible data augmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The duplicated latent representations, expanded according to the 'multiply_by' parameter.
    - Python dtype: `dict[str, torch.Tensor]`
- **`count`**
    - Comfy dtype: `INT`
    - The total count of latent representations after duplication, providing a straightforward way to understand the scale of the output dataset.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [LatentComposite](../../Comfy/Nodes/LatentComposite.md)



## Source code
```python
class DuplicateLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, latents: dict[str, Tensor], multiply_by: int):
        new_latents = latents.copy()
        full_latents = []
        for n in range(0, multiply_by):
            full_latents.append(new_latents["samples"])
        new_latents["samples"] = torch.cat(full_latents, dim=0)
        return (new_latents, new_latents["samples"].size(0),)

```
