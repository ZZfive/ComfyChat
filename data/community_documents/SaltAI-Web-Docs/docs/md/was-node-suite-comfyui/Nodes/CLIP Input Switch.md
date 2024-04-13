# CLIP Input Switch
## Documentation
- Class name: `CLIP Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

The WAS_CLIP_Input_Switch node is designed to toggle between two CLIP model inputs based on a boolean condition. It facilitates dynamic input selection within a workflow, enabling conditional logic and branching paths.
## Input types
### Required
- **`clip_a`**
    - The first CLIP model input. Acts as the default selection when the boolean condition is true.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
- **`clip_b`**
    - The second CLIP model input. Selected when the boolean condition is false.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
- **`boolean_number`**
    - A numeric boolean condition that determines which CLIP input to select. A value of 1 selects `clip_a`, while any other value selects `clip_b`.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`clip`**
    - Comfy dtype: `CLIP`
    - The selected CLIP model input based on the boolean condition.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_CLIP_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_a": ("CLIP",),
                "clip_b": ("CLIP",),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "clip_switch"

    CATEGORY = "WAS Suite/Logic"

    def clip_switch(self, clip_a, clip_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (clip_a, )
        else:
            return (clip_b, )

```
