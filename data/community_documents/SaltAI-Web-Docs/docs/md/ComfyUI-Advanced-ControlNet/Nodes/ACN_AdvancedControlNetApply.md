# Apply Advanced ControlNet 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_AdvancedControlNetApply`
- Category: `Adv-ControlNet 🛂🅐🅒🅝`
- Output node: `False`

This node applies advanced control networks to modify the conditioning of an image based on specified parameters, allowing for fine-tuned control over the image generation process. It supports a range of functionalities including the application of control nets, strength modulation, and optional parameters for further customization.
## Input types
### Required
- **`positive`**
    - The positive conditioning input, representing the desired attributes or features to enhance or maintain in the generated image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`negative`**
    - The negative conditioning input, representing the attributes or features to diminish or remove from the generated image.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`control_net`**
    - The control net to be applied, which dictates how the conditioning is modified.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`image`**
    - The input image to which the control net and conditioning modifications are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - A scalar value that determines the intensity of the control net's effect on the image conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - Specifies the starting percentage of the control net's effect, allowing for gradual application over a range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Specifies the ending percentage of the control net's effect, enabling fine-tuned control over its application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask_optional`**
    - An optional mask that can be applied to selectively target areas of the image for conditioning modification.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`timestep_kf`**
    - Optional keyframe information for time-stepping through the control net application, allowing for dynamic changes over time.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframeGroup`
- **`latent_kf_override`**
    - Optional override for latent keyframes, providing additional control over the image generation process.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframe`
- **`weights_override`**
    - Optional override for the control net weights, enabling customization of the control net's behavior.
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Python dtype: `ControlNetWeights`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The modified positive conditioning after applying the control net.
    - Python dtype: `List[Tuple[str, Dict]]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified negative conditioning after applying the control net.
    - Python dtype: `List[Tuple[str, Dict]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - KSampler (Efficient)
    - [ConditioningSetMask](../../Comfy/Nodes/ConditioningSetMask.md)



## Source code
```python
class AdvancedControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                "weights_override": ("CONTROL_NET_WEIGHTS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # copy, convert to advanced if needed, and set cond
                    c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent))
                    if is_advanced_controlnet(c_net):
                        # apply optional parameters and overrides, if provided
                        if timestep_kf is not None:
                            c_net.set_timestep_keyframes(timestep_kf)
                        if latent_kf_override is not None:
                            c_net.latent_keyframe_override = latent_kf_override
                        if weights_override is not None:
                            c_net.weights_override = weights_override
                        # verify weights are compatible
                        c_net.verify_all_weights()
                        # set cond hint mask
                        if mask_optional is not None:
                            mask_optional = mask_optional.clone()
                            # if not in the form of a batch, make it so
                            if len(mask_optional.shape) < 3:
                                mask_optional = mask_optional.unsqueeze(0)
                            c_net.set_cond_hint_mask(mask_optional)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

```
