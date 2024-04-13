# Latent Keyframe Interpolation 🛂🅐🅒🅝
## Documentation
- Class name: `LatentKeyframeTiming`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

The LatentKeyframeNode is designed for the creation, manipulation, and management of latent keyframes within a generative model's control network. It enables dynamic control over the generation process by adjusting keyframe properties such as timing, strength, and interpolation, facilitating the animation or transformation of generated content.
## Input types
### Required
- **`batch_index_from`**
    - Specifies the starting index for a batch of keyframes to be generated or modified, marking the beginning of the range within the sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_index_to_excl`**
    - Defines the exclusive end index for a batch of keyframes, indicating the end of the range within the sequence. This parameter helps in specifying the scope of keyframes to be affected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength_from`**
    - Specifies the starting strength value for the keyframes within the specified range, setting the initial intensity or effect level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_to`**
    - Defines the ending strength value for the keyframes, marking the final intensity or effect level within the specified range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`interpolation`**
    - Determines the method of interpolation to be used for calculating the strength values between the start and end of the specified range. This affects the gradual change in keyframe strength across the sequence.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`prev_latent_kf`**
    - References the previous group of latent keyframes, allowing for the addition or modification of keyframes within an existing sequence. This input is crucial for maintaining continuity in the animation or transformation process.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
- **`print_keyframes`**
    - A boolean flag that, when set to true, enables logging of keyframe information for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`LATENT_KF`**
    - Comfy dtype: `LATENT_KEYFRAME`
    - The modified or newly created group of latent keyframes, reflecting the applied changes. This output is essential for progressing the animation or transformation process with the desired dynamics.
    - Python dtype: `LatentKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LatentKeyframeInterpolationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index_from": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "batch_index_to_excl": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "strength_from": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_to": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": ([SI.LINEAR, SI.EASE_IN, SI.EASE_OUT, SI.EASE_IN_OUT], ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "print_keyframes": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                        batch_index_from: int,
                        strength_from: float,
                        batch_index_to_excl: int,
                        strength_to: float,
                        interpolation: str,
                        prev_latent_kf: LatentKeyframeGroup=None,
                        prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                        print_keyframes=False):

        if (batch_index_from > batch_index_to_excl):
            raise ValueError("batch_index_from must be less than or equal to batch_index_to.")

        if (batch_index_from < 0 and batch_index_to_excl >= 0):
            raise ValueError("batch_index_from and batch_index_to must be either both positive or both negative.")

        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        steps = batch_index_to_excl - batch_index_from
        diff = strength_to - strength_from
        if interpolation == SI.LINEAR:
            weights = np.linspace(strength_from, strength_to, steps)
        elif interpolation == SI.EASE_IN:
            index = np.linspace(0, 1, steps)
            weights = diff * np.power(index, 2) + strength_from
        elif interpolation == SI.EASE_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * (1 - np.power(1 - index, 2)) + strength_from
        elif interpolation == SI.EASE_IN_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

        for i in range(steps):
            keyframe = LatentKeyframe(batch_index_from + i, float(weights[i]))
            curr_latent_keyframe.add(keyframe)
        
        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"keyframe {keyframe.batch_index}:{keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

```
