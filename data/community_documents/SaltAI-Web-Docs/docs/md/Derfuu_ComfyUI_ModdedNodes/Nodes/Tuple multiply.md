# Tuple multiply
## Documentation
- Class name: `Tuple multiply`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The node multiplies each element within a tuple by a specified value, effectively scaling the tuple's elements.
## Input types
### Required
- **`Tuple`**
    - The tuple whose elements are to be scaled by the specified value.
    - Comfy dtype: `TUPLE`
    - Python dtype: `tuple`
- **`Value`**
    - The scalar value by which each element of the tuple is multiplied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - A new tuple with each of its elements scaled by the specified value.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MultiplyTupleBy:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "Tuple": ("TUPLE",),
                "Value": field.FLOAT,
            }
        }

    RETURN_TYPES = ("TUPLE",)
    FUNCTION = "MultiplyTuples"
    CATEGORY = TREE_TUPLES

    def MultiplyTuples(self, Tuple: tuple, Value: float):
        Tuple = (
            Tuple[0] * Value,
            Tuple[1] * Value
        )
        return (Tuple,)

```
