# Image Input Switch
## Documentation
- Class name: `Image Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node is designed to switch between two input images based on a boolean condition, effectively allowing conditional image selection within a workflow.
## Input types
### Required
- **`image_a`**
    - The first image option for the switch. This image will be selected if the boolean condition evaluates to true.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_b`**
    - The second image option for the switch. This image will be selected if the boolean condition evaluates to false.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`boolean_number`**
    - A numeric boolean condition that determines which image (image_a or image_b) is selected. A value of 1 selects image_a, while any other value selects image_b.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image selected based on the boolean condition.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ImageCompositeMasked](../../Comfy/Nodes/ImageCompositeMasked.md)



## Source code
```python
class WAS_Image_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def image_input_switch(self, image_a, image_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (image_a, )
        else:
            return (image_b, )

```
