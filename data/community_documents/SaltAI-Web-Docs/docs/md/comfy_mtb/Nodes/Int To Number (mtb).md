# Int To Number (mtb)
## Documentation
- Class name: `Int To Number (mtb)`
- Category: `mtb/number`
- Output node: `False`

The IntToNumber node is designed to seamlessly convert an integer value into a number format within the WAS Suite, ensuring compatibility and ease of use across different components.
## Input types
### Required
- **`int`**
    - This parameter represents the integer value to be converted into a number. It plays a crucial role in the conversion process, serving as the direct input that is transformed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - The output is the converted number, directly corresponding to the input integer value.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CM_NumberToFloat](../../ComfyMath/Nodes/CM_NumberToFloat.md)



## Source code
```python
class IntToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" INT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1e9,
                        "max": 1e9,
                        "step": 1,
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "int_to_number"
    CATEGORY = "mtb/number"

    def int_to_number(self, int):
        return (int,)

```
