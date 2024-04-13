# Conditioning (Combine)
## Documentation
- Class name: `ConditioningCombine`
- Category: `conditioning`
- Output node: `False`

This node combines two conditioning inputs into a single output, effectively merging their information.
## Input types
### Required
- **`conditioning_1`**
    - The first conditioning input to be combined. It plays an equal role with conditioning_2 in the combination process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
- **`conditioning_2`**
    - The second conditioning input to be combined. It is equally important as conditioning_1 in the merging process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The result of combining conditioning_1 and conditioning_2, encapsulating the merged information.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [ConditioningCombine](../../Comfy/Nodes/ConditioningCombine.md)
    - KSampler Adv. (Efficient)
    - [CR Conditioning Input Switch](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Conditioning Input Switch.md)
    - Attention couple
    - KSampler with Variations
    - [CR Module Pipe Loader](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Module Pipe Loader.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)



## Source code
```python
class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"

    CATEGORY = "conditioning"

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2, )

```
