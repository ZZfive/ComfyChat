# Power
## Documentation
- Class name: `Power`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The Power node calculates the result of raising a given value to the power of a specified exponent. It abstracts the mathematical power operation, enabling the dynamic computation of exponentiation within a data flow.
## Input types
### Required
- **`Value`**
    - The base value for the exponentiation operation. It serves as the foundation for the power calculation, determining the magnitude of the result.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Exponent`**
    - The exponent value in the exponentiation operation. It dictates the power to which the base value is raised, significantly influencing the outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of raising the base value to the power of the exponent. It represents the computed value of the exponentiation operation.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PowNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": field.FLOAT,
                "Exponent": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "pow"
    CATEGORY = TREE_MATH

    def pow(self, Value, Exponent):
        total = math.pow(Value, Exponent)
        return (total,)

```
