# [DEPR] Motion Model Settings (Simple) 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettingsSimple`
- Category: ``
- Output node: `False`

This node is designed to configure the motion model settings for the AnimateDiff process in a simplified manner. It allows users to adjust the scale of motion within the animation, providing a straightforward way to influence the animation's dynamics.
## Input types
### Required
- **`motion_pe_stretch`**
    - Determines the stretch of positional encoding in the motion model, affecting the range and intensity of motion effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask_motion_scale`**
    - An optional mask that can be applied to selectively adjust the motion scale across different parts of the animation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`min_motion_scale`**
    - Specifies the minimum scale for motion in the animation, setting a lower bound for how subtle the motion effects can be.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scale for motion in the animation, establishing an upper limit for the intensity of motion effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - The output is a configuration object for the AnimateDiff settings, encapsulating the adjustments made to the motion model.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettingsSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = ""  #"Animate Diff 🎭🅐🅓/① Gen1 nodes ①/motion settings/experimental"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, motion_pe_stretch: int,
                                  mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustPEGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch))
        motion_model_settings = AnimateDiffSettings(
            adjust_pe=adjust_pe,
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)

```
