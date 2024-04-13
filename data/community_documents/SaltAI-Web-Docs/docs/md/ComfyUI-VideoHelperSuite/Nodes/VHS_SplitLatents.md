# Split Latent Batch 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SplitLatents`
- Category: `Video Helper Suite 🎥🅥🅗🅢/latent`
- Output node: `False`

The VHS_SplitLatents node is designed to divide a batch of latents into two groups based on a specified index. This functionality is essential for managing and manipulating latent representations in various stages of video or image processing workflows.
## Input types
### Required
- **`latents`**
    - Represents the batch of latents to be split. It is crucial for determining how the latents are divided into two groups.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`split_index`**
    - Specifies the index at which the batch of latents is split into two groups. This index plays a pivotal role in how the latents are segmented.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`LATENT_A`**
    - Comfy dtype: `LATENT`
    - The first group of latents obtained after the split, up to the specified index.
    - Python dtype: `dict`
- **`A_count`**
    - Comfy dtype: `INT`
    - The count of latents in the first group, providing insight into the division's outcome.
    - Python dtype: `int`
- **`LATENT_B`**
    - Comfy dtype: `LATENT`
    - The second group of latents obtained after the split, starting from the specified index.
    - Python dtype: `dict`
- **`B_count`**
    - Comfy dtype: `INT`
    - The count of latents in the second group, indicating the division's result.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SplitLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": -99999999999}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT", "LATENT", "INT")
    RETURN_NAMES = ("LATENT_A", "A_count", "LATENT_B", "B_count")
    FUNCTION = "split_latents"

    def split_latents(self, latents: dict, split_index: int):
        latents = latents.copy()
        group_a = latents["samples"][:split_index]
        group_b = latents["samples"][split_index:]
        group_a_latent = {"samples": group_a}
        group_b_latent = {"samples": group_b}
        return (group_a_latent, group_a.size(0), group_b_latent, group_b.size(0))

```
