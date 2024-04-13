# [DEPR] Motion Model Settings (Advanced) 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettings`
- Category: ``
- Output node: `False`

This node is designed for configuring advanced motion model settings within the AnimateDiff framework. It allows for detailed customization of the motion model's behavior, including adjustments to positional encoding, attention strengths, and motion scaling, to fine-tune the animation effects.
## Input types
### Required
- **`pe_strength`**
    - Specifies the strength of the positional encoding adjustments, influencing the overall motion intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_strength`**
    - Determines the strength of the attention mechanism adjustments, affecting how motion is distributed across the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`other_strength`**
    - Controls additional strength parameters that may influence the motion model's behavior in less direct ways.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`motion_pe_stretch`**
    - Defines the stretch of positional encoding in the motion direction, allowing for more dynamic motion effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cap_initial_pe_length`**
    - Sets a cap on the initial length of positional encoding, which can help in controlling the motion's initial state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolate_pe_to_length`**
    - Specifies the length to which positional encoding should be interpolated, enabling smoother transitions in motion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`initial_pe_idx_offset`**
    - Determines the initial offset for positional encoding indices, affecting the starting point of motion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`final_pe_idx_offset`**
    - Sets the final offset for positional encoding indices, influencing the motion's end point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask_motion_scale`**
    - Provides a tensor to scale the motion mask, allowing for fine-grained control over how motion is applied.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`min_motion_scale`**
    - Specifies the minimum scale for motion, ensuring that motion effects do not fall below a certain threshold.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scale for motion, preventing overly exaggerated motion effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - Returns the configured motion model settings as an AD settings object, ready to be applied to the AnimateDiff framework.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettingsAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "other_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
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

    def get_motion_model_settings(self, pe_strength: float, attn_strength: float, other_strength: float,
                                  motion_pe_stretch: int,
                                  cap_initial_pe_length: int, interpolate_pe_to_length: int,
                                  initial_pe_idx_offset: int, final_pe_idx_offset: int,
                                  mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustPEGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch,
                             cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length,
                             initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset))
        motion_model_settings = AnimateDiffSettings(
            adjust_pe=adjust_pe,
            pe_strength=pe_strength,
            attn_strength=attn_strength,
            other_strength=other_strength,
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)

```
