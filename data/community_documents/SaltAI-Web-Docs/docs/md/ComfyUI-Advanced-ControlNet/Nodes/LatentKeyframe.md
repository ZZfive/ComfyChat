# Latent Keyframe 🛂🅐🅒🅝
## Documentation
- Class name: `LatentKeyframe`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

The LatentKeyframe node is designed for managing and manipulating latent keyframes within a generative model's control network. It focuses on loading and updating keyframe data based on given parameters, such as batch index and strength, to influence the generation process.
## Input types
### Required
- **`batch_index`**
    - Specifies the index of the batch for which the keyframe is being loaded or updated. It plays a crucial role in identifying the specific part of the generation process to be influenced.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Determines the intensity or influence of the keyframe on the generation process. A higher strength value indicates a stronger influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_latent_kf`**
    - An optional parameter that allows for the inclusion of a previously existing LatentKeyframeGroup, enabling the accumulation or modification of keyframes over time.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
## Output types
- **`LATENT_KF`**
    - Comfy dtype: `LATENT_KEYFRAME`
    - Returns an updated LatentKeyframeGroup, which includes the newly loaded or modified keyframe along with any existing keyframes.
    - Python dtype: `LatentKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LatentKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": -1000, "max": 1000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      batch_index: int,
                      strength: float,
                      prev_latent_kf: LatentKeyframeGroup=None,
                      prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                      ):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        keyframe = LatentKeyframe(batch_index, strength)
        prev_latent_keyframe.add(keyframe)
        return (prev_latent_keyframe,)

```
