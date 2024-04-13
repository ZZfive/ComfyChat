# Model Input Switch
## Documentation
- Class name: `Model Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

The Model Input Switch node is designed to select between two model inputs based on a boolean condition. It facilitates conditional logic in model processing pipelines, allowing for dynamic selection of models.
## Input types
### Required
- **`model_a`**
    - The first model input option. This model is selected if the boolean condition evaluates to true.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`model_b`**
    - The second model input option. This model is selected if the boolean condition evaluates to false.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`boolean_number`**
    - A numeric boolean condition that determines which model input to select. A value of 1 selects `model_a`, while any other value selects `model_b`.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The selected model input based on the boolean condition.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Model_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "model_b": ("MODEL",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "model_switch"

    CATEGORY = "WAS Suite/Logic"

    def model_switch(self, model_a, model_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (model_a, )
        else:
            return (model_b, )

```
