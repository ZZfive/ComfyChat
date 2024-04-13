# Tuple to floats
## Documentation
- Class name: `Tuple to floats`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The node converts a tuple of values into two separate float values, facilitating operations that require individual float inputs.
## Input types
### Required
- **`Tuple`**
    - The input tuple to be converted into two separate float values.
    - Comfy dtype: `TUPLE`
    - Python dtype: `tuple`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The second float value extracted from the input tuple.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Tuple2Float:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Tuple": ("TUPLE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    CATEGORY = TREE_TUPLES

    FUNCTION = 'get_tuple'

    def get_tuple(self, Tuple):
        return (Tuple[0], Tuple[1],)

```
