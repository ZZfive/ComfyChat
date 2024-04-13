# Divide
## Documentation
- Class name: `Divide`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The Divide node performs division between two floating-point numbers, outputting the result as a floating-point number. It abstracts the mathematical division operation, allowing for the division of a numerator by a denominator.
## Input types
### Required
- **`Numerator`**
    - Represents the dividend in the division operation. Its value directly influences the result of the division, serving as the number to be divided.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Denominator`**
    - Represents the divisor in the division operation. It determines how many times the numerator can be evenly divided, affecting the division's outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of dividing the Numerator by the Denominator, provided as a floating-point number.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DivideNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Numerator": field.FLOAT,
                "Denominator": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "divide"
    CATEGORY = TREE_MATH

    def divide(self, Numerator, Denominator):
        total = float(Numerator / Denominator)
        return (total,)

```
