# Control Net Model Input Switch
## Documentation
- Class name: `Control Net Model Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node is designed to switch between two control net models based on a boolean condition. It facilitates dynamic control net selection within a workflow, enabling conditional logic to dictate the control net model used for subsequent operations.
## Input types
### Required
- **`control_net_a`**
    - The first control net model option. This model is selected if the boolean condition evaluates to true.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`control_net_b`**
    - The second control net model option. This model is selected if the boolean condition evaluates to false.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`boolean_number`**
    - A numeric value that determines which control net model is selected. A value of 1 (or rounded to 1) selects `control_net_a`, while any other value selects `control_net_b`.
    - Comfy dtype: `NUMBER`
    - Python dtype: `float`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - The control net model selected based on the boolean condition.
    - Python dtype: `ControlNet`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Control_Net_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_a": ("CONTROL_NET",),
                "control_net_b": ("CONTROL_NET",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "control_net_switch"

    CATEGORY = "WAS Suite/Logic"

    def control_net_switch(self, control_net_a, control_net_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (control_net_a, )
        else:
            return (control_net_b, )

```
