# AnimateDiff Keyframe 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffKeyframe`
- Category: `Animate Diff 🎭🅐🅓`
- Output node: `False`

This node is designed for creating and managing keyframes within the AnimateDiff framework, allowing users to define and manipulate animation parameters such as start percentage, scale, and effects over time. It facilitates the dynamic generation of animation sequences by integrating new keyframes into existing sequences or creating new ones.
## Input types
### Required
- **`start_percent`**
    - Specifies the starting point of the keyframe as a percentage of the total animation duration. It determines when the keyframe's effects begin to apply, playing a crucial role in the timing of animations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_ad_keyframes`**
    - An optional collection of previously defined keyframes. This allows for the integration of the new keyframe into an existing sequence, enabling complex animations through the accumulation of keyframes.
    - Comfy dtype: `AD_KEYFRAMES`
    - Python dtype: `ADKeyframeGroup`
- **`scale_multival`**
    - Defines the scaling factor for the keyframe, which can be a single value or a tensor. This parameter influences the size or scale of the animated elements at this keyframe.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
- **`effect_multival`**
    - Specifies the effect intensity or parameters for the keyframe, which can also be a single value or a tensor. This affects how pronounced or subtle the animation effects are at this point.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
- **`inherit_missing`**
    - A boolean flag indicating whether to inherit unspecified values from previous keyframes. This facilitates smoother transitions and continuity in animations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`guarantee_steps`**
    - Determines the minimum number of steps to maintain the current keyframe's effects before transitioning to the next. This ensures a certain duration or stability for the keyframe's influence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`ad_keyframes`**
    - Comfy dtype: `AD_KEYFRAMES`
    - Returns the updated collection of keyframes, including the newly added keyframe, ready for use in further animation processing.
    - Python dtype: `ADKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ADKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_ad_keyframes": ("AD_KEYFRAMES", ),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: [float, torch.Tensor]=None, effect_multival: [float, torch.Tensor]=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        if not prev_ad_keyframes:
            prev_ad_keyframes = ADKeyframeGroup()
        prev_ad_keyframes = prev_ad_keyframes.clone()
        keyframe = ADKeyframe(start_percent=start_percent, scale_multival=scale_multival, effect_multival=effect_multival,
                              inherit_missing=inherit_missing, guarantee_steps=guarantee_steps)
        prev_ad_keyframes.add(keyframe)
        return (prev_ad_keyframes,)

```
