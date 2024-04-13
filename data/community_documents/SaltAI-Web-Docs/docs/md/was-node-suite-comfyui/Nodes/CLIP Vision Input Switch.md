# CLIP Vision Input Switch
## Documentation
- Class name: `CLIP Vision Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node provides a mechanism to switch between two CLIP vision models based on a boolean condition. It's part of the logic suite within the WAS node collection, enabling dynamic selection of models for further processing.
## Input types
### Required
- **`clip_vision_a`**
    - The first CLIP vision model to choose from. It plays a crucial role in determining the output based on the boolean condition.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `CLIP_VISION`
- **`clip_vision_b`**
    - The second CLIP vision model to choose from. This model is selected if the boolean condition evaluates to false.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `CLIP_VISION`
- **`boolean_number`**
    - A numeric boolean condition that determines which CLIP vision model is selected. A value of 1 selects the first model, while any other value selects the second.
    - Comfy dtype: `NUMBER`
    - Python dtype: `NUMBER`
## Output types
- **`clip_vision`**
    - Comfy dtype: `CLIP_VISION`
    - The selected CLIP vision model, determined by the boolean condition.
    - Python dtype: `CLIP_VISION`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_CLIP_Vision_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_a": ("CLIP_VISION",),
                "clip_vision_b": ("CLIP_VISION",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "clip_vision_switch"

    CATEGORY = "WAS Suite/Logic"

    def clip_vision_switch(self, clip_vision_a, clip_vision_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (clip_vision_a, )
        else:
            return (clip_vision_b)

```
