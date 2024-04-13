# Upscale Model Switch
## Documentation
- Class name: `Upscale Model Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

The Upscale Model Switch node is designed to toggle between two upscale models based on a boolean condition. It abstracts the decision-making process involved in selecting an appropriate upscale model for image processing tasks, thereby facilitating dynamic model switching in image upscaling pipelines.
## Input types
### Required
- **`upscale_model_a`**
    - The first upscale model option. This model is selected if the boolean condition evaluates to true.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `torch.nn.Module`
- **`upscale_model_b`**
    - The second upscale model option. This model is selected if the boolean condition evaluates to false.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `torch.nn.Module`
- **`boolean_number`**
    - A numeric value that determines which upscale model to select. A value of 1 selects the first model, while any other value selects the second model.
    - Comfy dtype: `NUMBER`
    - Python dtype: `float`
## Output types
- **`upscale_model`**
    - Comfy dtype: `UPSCALE_MODEL`
    - The selected upscale model based on the boolean condition.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Upscale_Model_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model_a": ("UPSCALE_MODEL",),
                "upscale_model_b": ("UPSCALE_MODEL",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "upscale_model_switch"

    CATEGORY = "WAS Suite/Logic"

    def upscale_model_switch(self, upscale_model_a, upscale_model_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (upscale_model_a, )
        else:
            return (upscale_model_b, )

```
