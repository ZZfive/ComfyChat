# Int To Bool (mtb)
## Documentation
- Class name: `Int To Bool (mtb)`
- Category: `mtb/number`
- Output node: `False`

The IntToBool node provides a straightforward conversion from an integer value to a boolean value, emphasizing simplicity and directness in data type transformation.
## Input types
### Required
- **`int`**
    - The integer input that will be converted to a boolean value. This parameter is essential for determining the output boolean value, where any non-zero integer will result in True, and zero will result in False.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`boolean`**
    - Comfy dtype: `BOOLEAN`
    - The output is a boolean representation of the input integer, where non-zero values are True, and zero is False.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - ReActorFaceSwap



## Source code
```python
class IntToBool:
    """Basic int to bool conversion"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "int_to_bool"
    CATEGORY = "mtb/number"

    def int_to_bool(self, int):
        return (bool(int),)

```
