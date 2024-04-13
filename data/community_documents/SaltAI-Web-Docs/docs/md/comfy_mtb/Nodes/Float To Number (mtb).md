# Float To Number (mtb)
## Documentation
- Class name: `Float To Number (mtb)`
- Category: `mtb/number`
- Output node: `False`

Node addon for the WAS Suite designed to convert a floating-point number into a more general number format, facilitating compatibility and further processing within the suite.
## Input types
### Required
- **`float`**
    - The floating-point number to be converted. This input is crucial as it directly determines the output number, ensuring the conversion process aligns with the node's purpose.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - The converted number from the floating-point input, enabling further numerical operations or compatibility within the suite.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloatToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" FLOAT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": (
                    "FLOAT",
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
    FUNCTION = "float_to_number"
    CATEGORY = "mtb/number"

    def float_to_number(self, float):
        return (float,)

        return (int,)

```
