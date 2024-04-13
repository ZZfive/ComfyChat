# Square root
## Documentation
- Class name: `Square root`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The SquareRootNode calculates the square root of a given value, providing a straightforward way to perform this mathematical operation.
## Input types
### Required
- **`Value`**
    - The 'Value' parameter represents the number for which the square root is to be calculated. It is crucial for determining the output of the node.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - This output represents the square root value of the input number.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SquareRootNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    FUNCTION = "sqrt"
    CATEGORY = TREE_MATH

    def sqrt(self, Value):
        total = math.sqrt(Value)
        return (total, -total,)

```
