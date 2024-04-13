# Multiply
## Documentation
- Class name: `Multiply`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The Multiply node performs a multiplication operation on two input values, producing a single output that represents the product of the inputs. This node is fundamental in mathematical operations, allowing for the scaling and combination of values in a computational graph.
## Input types
### Required
- **`Value_A`**
    - Represents one of the multiplicands in the multiplication operation. Its value directly influences the magnitude of the output product.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Value_B`**
    - Represents the second multiplicand in the multiplication operation. Altering this value changes the output product, showcasing its direct impact on the multiplication result.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output of the Multiply node is a single float value that represents the product of the two input values.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - floatToInt _O



## Source code
```python
class MultiplyNode:
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
    FUNCTION = "multiply"
    CATEGORY = TREE_MATH

    def multiply(self, Value_A, Value_B):
        total = float(Value_A * Value_B)
        return (total,)

```
