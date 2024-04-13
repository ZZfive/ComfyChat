# Select Every Nth Latent 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthLatent`
- Category: `Video Helper Suite 🎥🅥🅗🅢/latent`
- Output node: `False`

This node is designed to filter a batch of latents by selecting every Nth latent from the batch. It's useful for thinning out dense latent sequences to reduce processing load or to select a subset of latents for specific purposes.
## Input types
### Required
- **`latents`**
    - The input latents to be filtered. This parameter is crucial for determining which latents will be considered for selection.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`select_every_nth`**
    - Determines the interval at which latents are selected from the input batch. A value of 1 means every latent is selected, while higher values thin out the selection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The filtered subset of latents after selecting every Nth latent.
    - Python dtype: `dict`
- **`count`**
    - Comfy dtype: `INT`
    - The total count of latents selected and returned by the node.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "select_latents"

    def select_latents(self, latents: dict, select_every_nth: int):
        sub_latents = latents.copy()["samples"][0::select_every_nth]
        return ({"samples": sub_latents}, sub_latents.size(0))

```
