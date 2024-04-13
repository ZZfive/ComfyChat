# Tuple
## Documentation
- Class name: `Tuple`
- Category: `Derfuu_Nodes/Tuples`
- Output node: `False`

The Tuple node is designed to create a tuple from two float values, with an option to round the values in various ways (no rounding, standard rounding, ceiling, or floor). This functionality allows for precise control over the numerical data being grouped into a tuple, catering to different rounding needs.
## Input types
### Required
- **`Value_A`**
    - Specifies the first float value to be included in the tuple. Its value directly influences the tuple's first element.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Value_B`**
    - Determines the second float value to be included in the tuple. Its value directly influences the tuple's second element.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`round_`**
    - Controls the rounding method applied to the float values before they are grouped into a tuple. Options include no rounding, standard rounding, ceiling, or floor, affecting the tuple's numerical precision.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`tuple`**
    - Comfy dtype: `TUPLE`
    - Outputs a tuple composed of the two input float values, potentially modified according to the specified rounding method.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Tuple:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": field.FLOAT,
                "Value_B": field.FLOAT,
                "round_": (["No", "Yes", "Ceil", "Floor"],),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    CATEGORY = TREE_TUPLES

    FUNCTION = 'get_tuple'

    def get_tuple(self, Value_A: float = 0, Value_B: float = 0, round_: str = "No"):
        if round_ == "No":
            return ((Value_A, Value_B),)
        elif round_ == "Yes":
            return ((round(Value_A), round(Value_B)),)
        elif round_ == "Ceil":
            return ((math.ceil(Value_A), math.ceil(Value_B)),)
        elif round_ == "Floor":
            return ((math.floor(Value_A), math.floor(Value_B)),)
        # py 3.10+
        # match round:
        #     case "No":
        #         return ((Value_A, Value_B),)
        #     case "Yes":
        #         return ((int(Value_A), int(Value_B)),)
        #     case "Ceil":
        #         return ((math.ceil(Value_A), math.ceil(Value_B)),)
        #     case "Floor":
        #         return ((math.floor(Value_A), math.floor(Value_B)),)

```
