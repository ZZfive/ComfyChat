# Ceil
## Documentation
- Class name: `Ceil`
- Category: `Derfuu_Nodes/Functions/Converters`
- Output node: `False`

The CeilNode is designed to round up a given floating-point number to the nearest integer. It abstracts the mathematical ceiling function, ensuring that any input value is elevated to its immediate higher whole number.
## Input types
### Required
- **`Value`**
    - The 'Value' parameter represents the floating-point number that is to be rounded up to the nearest integer. It is crucial for determining the output of the CeilNode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The output is the nearest integer that is greater than or equal to the input floating-point number.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CeilNode:
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
        total = int(math.ceil(Value))
        return (total,)

```
