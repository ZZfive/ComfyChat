# [DEPR] Motion Model Settings 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettings_Release`
- Category: ``
- Output node: `False`

This node is designed for configuring motion model settings within the AnimateDiff framework, specifically targeting the release version. It allows for the adjustment of parameters related to the motion and appearance of animations, facilitating the creation of dynamic and visually appealing content.
## Input types
### Required
- **`min_motion_scale`**
    - Sets the minimum scaling factor for motion, ensuring a baseline level of motion intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scaling factor for motion, capping the intensity of movements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask_motion_scale`**
    - Applies a scaling factor to the motion mask, enabling finer control over the motion's intensity and distribution.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - Returns the configured motion model settings, encapsulating adjustments to various parameters for fine-tuning the animation's motion and appearance.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = ""  #"Animate Diff 🎭🅐🅓/① Gen1 nodes ①/motion settings"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        motion_model_settings = AnimateDiffSettings(
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)

```
