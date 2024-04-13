# Floor
## Documentation
- Class name: `Floor`
- Category: `Derfuu_Nodes/Functions/Converters`
- Output node: `False`

The Floor node rounds a given floating-point value down to the nearest integer, effectively truncating any fractional part.
## Input types
### Required
- **`Value`**
    - Specifies the floating-point value to be rounded down. This parameter determines the exact integer value that will be returned by truncating any fractional part of the input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer result obtained by rounding down the input floating-point value.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloorNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": field.FLOAT,
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_value"
    CATEGORY = TREE_CONVERTERS

    def get_value(self, Value):
        total = int(math.floor(Value))
        return (total,)

```
