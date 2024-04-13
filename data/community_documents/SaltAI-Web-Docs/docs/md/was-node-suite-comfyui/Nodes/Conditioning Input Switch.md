# Conditioning Input Switch
## Documentation
- Class name: `Conditioning Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node is designed to switch between two conditioning inputs based on a boolean value, effectively allowing conditional logic to be applied to the flow of conditioning data within a workflow.
## Input types
### Required
- **`conditioning_a`**
    - The first conditioning input to be considered for the switch. It plays a crucial role in determining the output based on the boolean condition.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
- **`conditioning_b`**
    - The second conditioning input to be considered for the switch. It serves as an alternative to the first input, depending on the boolean condition.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
- **`boolean_number`**
    - A numeric boolean value (typically 1 or 0) that determines which conditioning input to pass through. It acts as the condition for the switch.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The selected conditioning input based on the boolean condition. It represents the outcome of the conditional logic applied.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Conditioning_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "conditioning_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def conditioning_input_switch(self, conditioning_a, conditioning_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (conditioning_a, )
        else:
            return (conditioning_b, )

```
