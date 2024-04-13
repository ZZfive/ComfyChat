# Tuple swap
## Documentation
- Class name: `Tuple swap`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The node is designed to swap the elements within a tuple, effectively reversing their order. It abstracts the process of reordering tuple elements, making it straightforward to invert the positions of items within a tuple.
## Input types
### Required
- **`Tuple`**
    - The tuple whose elements are to be swapped. This operation reverses the order of the elements, transforming the tuple from (a, b) to (b, a).
    - Comfy dtype: `TUPLE`
    - Python dtype: `tuple`
## Output types
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - The resulting tuple after the elements have been swapped, with the original second element now first, and the original first element now second.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FlipTuple:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Tuple": ("TUPLE",)
            }
        }

    RETURN_TYPES = ("TUPLE",)
    FUNCTION = "flip"
    CATEGORY = TREE_TUPLES

    def flip(self, Tuple):
        return ((Tuple[1], Tuple[0]),)

```
