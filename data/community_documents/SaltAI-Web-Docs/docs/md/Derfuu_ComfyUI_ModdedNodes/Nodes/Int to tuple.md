# Int to tuple
## Documentation
- Class name: `Int to tuple`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The node transforms two integer values into a tuple, encapsulating them in a structured format suitable for further processing or manipulation.
## Input types
### Required
- **`Value_A`**
    - Specifies the first integer value to be included in the tuple.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`Value_B`**
    - Specifies the second integer value to be included in the tuple.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - Outputs a tuple containing the two input integer values.
    - Python dtype: `Tuple[int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Int2Tuple:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": field.INT,
                "Value_B": field.INT,
            }
        }

    RETURN_TYPES = ("TUPLE",)
    CATEGORY = TREE_TUPLES

    FUNCTION = 'get_tuple'

    def get_tuple(self, Value_A=0, Value_B=0):
        return ((Value_A, Value_B),)

```
