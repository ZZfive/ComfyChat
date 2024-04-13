# Cosines
## Documentation
- Class name: `Cosines`
- Category: `Derfuu_Nodes/Math/Trigonometry`
- Output node: `False`

The Cosines node calculates the cosine of a given angle, which can be specified in either radians or degrees. It also supports computing the arccosine of a value, returning the angle whose cosine is the given number.
## Input types
### Required
- **`value`**
    - The value of the angle for which the cosine is to be calculated, or the value whose arccosine is to be found. It plays a crucial role in determining the output of the node.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`type_`**
    - Specifies whether the input value is in radians ('RAD') or degrees ('DEG'), affecting how the value is processed before calculating the cosine or arccosine.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`arcCos`**
    - A boolean flag that, when set to True, indicates the node should compute the arccosine of the input value instead of the cosine.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of the cosine or arccosine calculation, depending on the input parameters.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CosNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "value": field.FLOAT,
                "type_": (["RAD", "DEG"],),
                "arcCos": ([False, True],)
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_value"
    CATEGORY = TREE_TRIGONOMETRY

    def get_value(self, value, type_="RAD", arcCos=False):
        if type_ == "DEG":
            value = math.radians(value)
        if arcCos == True:
            value = math.acos(value)
        else:
            value = math.cos(value)
        return (value,)

```
