# Subtract
## Documentation
- Class name: `Subtract`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The SubtractNode is designed for performing subtraction operations between two floating-point values. It abstracts the mathematical subtraction process, allowing users to easily compute the difference between two numbers.
## Input types
### Required
- **`Value_A`**
    - Represents the minuend in the subtraction operation, from which Value_B will be subtracted.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Value_B`**
    - Represents the subtrahend in the subtraction operation, which will be subtracted from Value_A.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of subtracting Value_B from Value_A, returned as a floating-point number.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SubtractNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": field.FLOAT,
                "Value_B": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "sub"
    CATEGORY = TREE_MATH

    def sub(self, Value_A, Value_B):
        total = float(Value_A - Value_B)
        return (total,)

```
