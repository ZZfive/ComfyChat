# Int to float
## Documentation
- Class name: `Int to float`
- Category: `Derfuu_Nodes/Functions/Converters`
- Output node: `False`

The node is designed to convert an integer value to its floating-point representation, facilitating operations that require floating-point arithmetic.
## Input types
### Required
- **`Value`**
    - This parameter represents the integer value to be converted into a floating-point number, enabling precise calculations and operations that require decimal precision.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output is the floating-point representation of the input integer, allowing for more nuanced and precise calculations.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Int2Float:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": field.INT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_value"
    CATEGORY = TREE_CONVERTERS

    def get_value(self, Value):
        return (float(Value),)

```
