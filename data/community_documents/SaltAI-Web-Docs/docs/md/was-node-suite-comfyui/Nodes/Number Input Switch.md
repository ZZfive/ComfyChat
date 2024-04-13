# Number Input Switch
## Documentation
- Class name: `Number Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

The WAS_Number_Input_Switch node is designed to select between two numerical inputs based on a boolean condition. It provides a flexible mechanism for conditional logic in numerical data processing workflows.
## Input types
### Required
- **`number_a`**
    - The first numerical input option for the switch. Its selection is determined by the boolean_number input, influencing the node's output based on conditional logic.
    - Comfy dtype: `NUMBER`
    - Python dtype: `float | int`
- **`number_b`**
    - The second numerical input option for the switch. Its selection is influenced by the boolean_number input, affecting the node's output through conditional logic.
    - Comfy dtype: `NUMBER`
    - Python dtype: `float | int`
- **`boolean_number`**
    - A numerical value that determines which input (number_a or number_b) to pass through. Typically, a non-zero value selects number_a, while zero selects number_b, directly impacting the node's execution and results.
    - Comfy dtype: `NUMBER`
    - Python dtype: `float | int`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - Outputs the selected number as is, providing versatility in subsequent processing.
    - Python dtype: `float | int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - Outputs the selected number as a float, enabling flexible numerical operations.
    - Python dtype: `float`
- **`int`**
    - Comfy dtype: `INT`
    - Outputs the selected number as an integer, facilitating operations requiring discrete values.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Number_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_a": ("NUMBER",),
                "number_b": ("NUMBER",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")
    FUNCTION = "number_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def number_input_switch(self, number_a, number_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (number_a, float(number_a), int(number_a))
        else:
            return (number_b, float(number_b), int(number_b))

```
