# Integer
## Documentation
- Class name: `Integer`
- Category: `Derfuu_Nodes/Variables`
- Output node: `False`

The Integer node is designed to convert a floating-point value to an integer. It serves as a basic type conversion node within a broader system, facilitating the manipulation and processing of numerical data by ensuring type consistency.
## Input types
### Required
- **`Value`**
    - This parameter represents the floating-point value to be converted into an integer. It plays a crucial role in the node's operation by determining the resultant integer value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - This output represents the integer value obtained from converting the input floating-point value.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CLIPTextEncodeSDXL](../../Comfy/Nodes/CLIPTextEncodeSDXL.md)
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - [Image Resize](../../was-node-suite-comfyui/Nodes/Image Resize.md)
    - KSampler Adv. (Efficient)



## Source code
```python
class IntegerNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("FLOAT", {
                        "default": 1,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1
                    },
                )
            },
        }

    RETURN_TYPES = ("INT",)
    CATEGORY = TREE_VARIABLE
    FUNCTION = "get_value"

    def get_value(self, Value):
        return (int(Value),)

```
