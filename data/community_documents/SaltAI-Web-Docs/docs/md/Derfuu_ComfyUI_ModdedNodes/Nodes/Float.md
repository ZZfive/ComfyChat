# Float
## Documentation
- Class name: `Float`
- Category: `Derfuu_Nodes/Variables`
- Output node: `False`

The Float node is designed to return a floating-point value as specified by the user. It serves as a basic variable node that allows for the specification of decimal values within a flow.
## Input types
### Required
- **`Value`**
    - Specifies the floating-point value to be returned by the node. This parameter allows the user to input a decimal value, which the node will directly output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - Outputs the floating-point value specified by the user in the input.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - ezMath
    - KSampler Adv. (Efficient)
    - Reroute
    - workflow/IP Adapter full bundle



## Source code
```python
class FloatNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = TREE_VARIABLE
    FUNCTION = "get_value"

    def get_value(self, Value):
        return (Value,)

```
