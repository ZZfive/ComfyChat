# Tuple to ints
## Documentation
- Class name: `Tuple to ints`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The node converts a tuple of values into a pair of integer values. It is designed to facilitate the transformation of data types within a workflow, specifically targeting the conversion of tuple elements into integers for further processing or analysis.
## Input types
### Required
- **`Tuple`**
    - The input tuple containing elements to be converted into integers. This parameter is crucial for specifying the data that will undergo type conversion.
    - Comfy dtype: `TUPLE`
    - Python dtype: `Tuple[Any, Any]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The second integer value extracted and converted from the input tuple.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Tuple2Int:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Tuple": ("TUPLE",),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    CATEGORY = TREE_TUPLES

    FUNCTION = 'get_tuple'

    def get_tuple(self, Tuple):
        return (int(Tuple[0]), int(Tuple[1]),)

```
