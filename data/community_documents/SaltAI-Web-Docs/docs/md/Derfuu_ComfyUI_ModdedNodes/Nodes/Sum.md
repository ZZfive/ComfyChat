# Sum
## Documentation
- Class name: `Sum`
- Category: `Derfuu_Nodes/Math`
- Output node: `False`

The Sum node is designed to perform a simple arithmetic operation of addition, taking two numerical values as input and returning their sum. This node simplifies the process of adding two numbers, abstracting the underlying mathematical operation.
## Input types
### Required
- **`Value_A`**
    - Represents the first value to be added. Its role is crucial as it serves as one of the two operands in the addition operation, directly influencing the result.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Value_B`**
    - Serves as the second operand in the addition operation. Its value is added to the first operand (Value_A), and together they determine the sum output of the node.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of adding the two input values (Value_A and Value_B). It represents the sum of the inputs.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SumNode:
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
    FUNCTION = "sum"
    CATEGORY = TREE_MATH

    def sum(self, Value_A, Value_B):
        total = float(Value_A + Value_B)
        return (total,)

```
