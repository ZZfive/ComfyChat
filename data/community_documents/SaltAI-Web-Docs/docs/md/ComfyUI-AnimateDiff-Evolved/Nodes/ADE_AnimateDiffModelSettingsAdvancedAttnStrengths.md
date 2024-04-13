# [DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettingsAdvancedAttnStrengths`
- Category: ``
- Output node: `False`

This node is designed to configure advanced attention strengths within the AnimateDiff framework. It allows for fine-tuning of the animation process by adjusting various attention-related parameters, providing a more detailed control over the motion model's behavior.
## Input types
### Required
- **`pe_strength`**
    - Defines the strength of positional encoding adjustments, influencing the overall animation effect by modifying how position information is factored into the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_strength`**
    - Sets the base strength for attention mechanisms, affecting how strongly the model's attention layers contribute to the animation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_q_strength`**
    - Adjusts the strength specifically for the query component of the attention mechanism, offering nuanced control over how query vectors influence the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_k_strength`**
    - Modifies the strength for the key component of the attention mechanism, affecting how key vectors are weighted in the animation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_v_strength`**
    - Alters the strength for the value component of the attention mechanism, impacting how value vectors contribute to the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_out_weight_strength`**
    - Controls the strength of the output weights in the attention mechanism, fine-tuning the output layer's influence on the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_out_bias_strength`**
    - Adjusts the strength of the output bias in the attention mechanism, allowing for detailed control over the bias's effect on the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`other_strength`**
    - Adjusts the strength of other unspecified model components, offering a way to fine-tune parts of the model not directly related to positional encoding or attention.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`motion_pe_stretch`**
    - Specifies the extent to which positional encoding is stretched, impacting the animation's motion characteristics.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cap_initial_pe_length`**
    - Caps the initial length of positional encoding, potentially limiting the range of motion in the animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolate_pe_to_length`**
    - Determines the target length for positional encoding interpolation, affecting the smoothness and range of the animation's motion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`initial_pe_idx_offset`**
    - Offsets the starting index for positional encoding, allowing for adjustments in the animation's starting position.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`final_pe_idx_offset`**
    - Offsets the ending index for positional encoding, enabling modifications to the animation's ending position.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask_motion_scale`**
    - Applies a mask to scale the motion, providing a method to selectively enhance or reduce motion in specific areas of the animation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`min_motion_scale`**
    - Sets the minimum scale for motion, ensuring that the animation's motion does not fall below a certain threshold.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scale for motion, capping the intensity of the animation's motion to prevent it from exceeding a desired level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - Outputs the configured motion model settings, encapsulating all the adjustments made to attention strengths and other parameters for use in the animation process.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettingsAdvancedAttnStrengths:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_q_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_k_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_weight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_bias_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
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

    def get_motion_model_settings(self, pe_strength: float, attn_strength: float,
                                  attn_q_strength: float,
                                  attn_k_strength: float,
                                  attn_v_strength: float,
                                  attn_out_weight_strength: float,
                                  attn_out_bias_strength: float,
                                  other_strength: float,
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
            attn_q_strength=attn_q_strength,
            attn_k_strength=attn_k_strength,
            attn_v_strength=attn_v_strength,
            attn_out_weight_strength=attn_out_weight_strength,
            attn_out_bias_strength=attn_out_bias_strength,
            other_strength=other_strength,
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)

```
