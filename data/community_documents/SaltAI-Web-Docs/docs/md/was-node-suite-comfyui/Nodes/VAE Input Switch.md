# VAE Input Switch
## Documentation
- Class name: `VAE Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node provides a mechanism to switch between two VAE (Variational Autoencoder) models based on a boolean condition. It's part of the WAS Suite's logic category, facilitating dynamic model selection in workflows.
## Input types
### Required
- **`vae_a`**
    - The first VAE model to choose from. It plays a crucial role in determining the output based on the boolean condition.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`vae_b`**
    - The second VAE model to choose from. This model is selected if the boolean condition evaluates to false.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`boolean_number`**
    - A numeric boolean condition that determines which VAE model (vae_a or vae_b) is selected. A value of 1 selects vae_a, while any other value selects vae_b.
    - Comfy dtype: `NUMBER`
    - Python dtype: `NUMBER`
## Output types
- **`vae`**
    - Comfy dtype: `VAE`
    - The selected VAE model based on the boolean condition.
    - Python dtype: `VAE`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_VAE_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_a": ("VAE",),
                "vae_b": ("VAE",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "vae_switch"

    CATEGORY = "WAS Suite/Logic"

    def vae_switch(self, vae_a, vae_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (vae_a, )
        else:
            return (vae_b, )

```
