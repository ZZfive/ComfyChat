# Float Constant
## Documentation
- Class name: `FloatConstant`
- Category: `KJNodes`
- Output node: `False`

The FloatConstant node provides a mechanism to define a floating-point constant value within a computational graph. It allows for the specification of a constant floating-point value that can be used as an input to other nodes or operations, facilitating the incorporation of fixed numerical values into dynamic computational workflows.
## Input types
### Required
- **`value`**
    - Specifies the floating-point constant value to be used within the node. This value serves as a fixed numerical input that influences subsequent computations or operations in the workflow.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`value`**
    - Comfy dtype: `FLOAT`
    - Returns the specified floating-point constant value, enabling its use in further computations or operations within the computational graph.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloatConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001}),
        },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "KJNodes"

    def get_value(self, value):
        return (value,)

```
