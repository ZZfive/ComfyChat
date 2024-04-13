# Lora Input Switch
## Documentation
- Class name: `Lora Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

This node is designed to switch between two sets of inputs (model and clip pairs) based on a boolean condition. It facilitates dynamic selection in workflows that require conditional logic to determine the appropriate inputs for further processing.
## Input types
### Required
- **`model_a`**
    - The first model input for the switch. It represents one of the potential choices based on the boolean condition.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip_a`**
    - The first clip input paired with `model_a`. It complements the model input in the selection process.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`model_b`**
    - The second model input for the switch. It serves as an alternative choice to `model_a` based on the boolean condition.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip_b`**
    - The second clip input paired with `model_b`. It provides an alternative to `clip_a` in the selection process.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`boolean_number`**
    - A numeric boolean condition that determines which set of inputs (A or B) to select. A value of 1 selects the A set, while any other value selects the B set.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The selected model input based on the boolean condition.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The selected clip input paired with the chosen model.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Lora_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL",),
                "clip_a": ("CLIP",),
                "model_b": ("MODEL",),
                "clip_b": ("CLIP",),
                "boolean_number": ("NUMBER",),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "lora_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def lora_input_switch(self, model_a, clip_a, model_b, clip_b, boolean_number=1):
        if int(round(boolean_number)) == 1:
            return (model_a, clip_a)
        else:
            return (model_b, clip_b)

```
