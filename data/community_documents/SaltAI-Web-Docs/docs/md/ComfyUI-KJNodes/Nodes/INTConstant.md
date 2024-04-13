# INT Constant
## Documentation
- Class name: `INTConstant`
- Category: `KJNodes`
- Output node: `False`

Provides a constant integer value as specified by the user. This node is designed to supply fixed integer values to other nodes or processes within a workflow, facilitating operations that require specific, unchanging numerical inputs.
## Input types
### Required
- **`value`**
    - The integer value to be returned by the node. This parameter allows the user to specify the exact integer value they wish to use as a constant throughout their workflow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`value`**
    - Comfy dtype: `INT`
    - The constant integer value specified by the user.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - SetNode
    - [ImageGrabPIL](../../ComfyUI-KJNodes/Nodes/ImageGrabPIL.md)
    - [PixelPerfectResolution](../../comfyui_controlnet_aux/Nodes/PixelPerfectResolution.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)



## Source code
```python
class INTConstant:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"

    CATEGORY = "KJNodes"

    def get_value(self, value):
        return (value,)

```
