# Absolute value
## Documentation
- Class name: `Absolute value`
- Category: `Derfuu_Nodes/Functions/Converters`
- Output node: `False`

The node calculates the absolute value of a given input. It can optionally return the negative absolute value based on a boolean flag.
## Input types
### Required
- **`Value`**
    - Represents the numerical value for which the absolute value is to be calculated. It is crucial for determining the magnitude of the input regardless of its sign.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`negative_out`**
    - A boolean flag that, when set to True, causes the node to return the negative of the absolute value of the input. This allows for flexibility in handling negative values.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output is the absolute value of the input number. If the 'negative_out' flag is set to True, it returns the negative absolute value instead.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ABSNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "Value": field.FLOAT,
                "negative_out": ([False, True],)
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "abs_val"
    CATEGORY = TREE_CONVERTERS

    def abs_val(self, Value, Get_negative):
        if Get_negative:
            return (-abs(Value),)
        return (abs(Value),)

```
