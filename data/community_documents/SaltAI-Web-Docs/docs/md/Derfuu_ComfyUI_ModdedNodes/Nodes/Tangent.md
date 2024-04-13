# Tangent
## Documentation
- Class name: `Tangent`
- Category: `Derfuu_Nodes/Math/Trigonometry`
- Output node: `False`

The Tangent node calculates the tangent of a given angle, which can be specified in either radians or degrees. It also offers the option to compute the arctangent of the input value, providing flexibility in trigonometric calculations.
## Input types
### Required
- **`value`**
    - Represents the angle for which the tangent or arctangent is to be calculated. The choice between tangent and arctangent computation is determined by the 'arcTan' parameter.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`type_`**
    - Specifies the unit of the input angle: radians ('RAD') or degrees ('DEG'). This allows for easy conversion and ensures the calculation is performed in the correct unit.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[str]`
- **`arcTan`**
    - A boolean flag that, when set to True, switches the operation from calculating the tangent to calculating the arctangent of the input value. This provides an additional trigonometric operation within the same node.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `Tuple[bool]`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The result of the tangent or arctangent calculation, depending on the input parameters.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class tgNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "value": field.FLOAT,
                "type_": (["RAD", "DEG"],),
                "arcTan": ([False, True],)
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_value"
    CATEGORY = TREE_TRIGONOMETRY

    def get_value(self, value, type_="RAD", arcTan=False):
        if type_ == "DEG":
            value = math.radians(value)
        if arcTan == True:
            value = math.atan(value)
        else:
            value = math.tan(value)
        return (value,)

```
