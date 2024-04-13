# Get latent size
## Documentation
- Class name: `Get latent size`
- Category: `Derfuu_Nodes/Functions`
- Output node: `False`

This node is designed to calculate the dimensions of a latent representation. It provides the width and height of the latent space, optionally adjusting the size based on whether the original dimensions are requested or a scaled version is preferred.
## Input types
### Required
- **`latent`**
    - The latent representation for which the size is to be calculated. This parameter is crucial for determining the dimensions of the latent space.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, Any]`
- **`original`**
    - A boolean flag indicating whether the original dimensions of the latent space are requested (True) or a scaled version is preferred (False). This affects the final size calculation.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The height of the latent representation.
    - Python dtype: `int`
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - A tuple containing both the width and height of the latent representation.
    - Python dtype: `Tuple[int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetLatentSize:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "original": ([False, True],),
            }
        }

    RETURN_TYPES = ("INT", "INT", "TUPLE",)
    CATEGORY = TREE_FUNCTIONS

    FUNCTION = 'get_size'

    def get_size(self, latent, original):
        size = sizes.get_latent_size(latent, original)
        return (size[0], size[1], size,)

```
