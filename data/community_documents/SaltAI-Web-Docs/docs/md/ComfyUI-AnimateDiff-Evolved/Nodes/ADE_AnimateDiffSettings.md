# AnimateDiff Settings 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffSettings`
- Category: `Animate Diff 🎭🅐🅓/ad settings`
- Output node: `False`

This node is designed to configure and apply specific settings for the AnimateDiff process, allowing for adjustments to the animation dynamics through parameters like PE adjustment. It serves as a foundational element in customizing the behavior and output of AnimateDiff animations.
## Input types
### Optional
- **`pe_adjust`**
    - Specifies the type of positional encoding (PE) adjustment to be applied, influencing the animation's dynamics and overall appearance.
    - Comfy dtype: `PE_ADJUST`
    - Python dtype: `AdjustPEGroup`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - The configured AnimateDiff settings, ready to be utilized in subsequent animation processes.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "pe_adjust": ("PE_ADJUST",),
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings"
    FUNCTION = "get_ad_settings"

    def get_ad_settings(self, pe_adjust: AdjustPEGroup=None):
        return (AnimateDiffSettings(adjust_pe=pe_adjust),)

```
