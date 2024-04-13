# Latent Input Switch
## Documentation
- Class name: `Latent Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node is designed to switch between two latent inputs based on a boolean condition, effectively allowing for conditional selection between different latent representations.
## Input types
### Required
- **`latent_a`**
    - The first latent input to consider for selection. It plays a crucial role in determining the output based on the boolean condition.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`latent_b`**
    - The second latent input to consider for selection. It serves as an alternative to the first latent input, depending on the boolean condition.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`boolean_number`**
    - A numeric boolean condition that determines which latent input to select. A value of 1 selects the first latent input, while any other value selects the second.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The selected latent input based on the boolean condition. It represents the conditional choice between the two provided latent inputs.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Latent_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "latent_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def latent_input_switch(self, latent_a, latent_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (latent_a, )
        else:
            return (latent_b, )

```
