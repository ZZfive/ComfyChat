# Keyframing Perlin Settings
## Documentation
- Class name: `OPACPerlinSettings`
- Category: `SALT/Animation/Keyframing`
- Output node: `False`

This node is designed for configuring Perlin noise sampling parameters within the OPAC node. It allows for the customization of noise characteristics to achieve varied and dynamic animation effects.
## Input types
### Required
- **`zoom_octaves`**
    - Specifies the number of octaves for the zoom effect in Perlin noise, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`zoom_persistence`**
    - Determines the persistence value for the zoom effect, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_lacunarity`**
    - Controls the lacunarity for the zoom effect, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_repeat`**
    - Sets the repeat interval for the zoom effect in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle_octaves`**
    - Specifies the number of octaves for the angle effect in Perlin noise, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle_persistence`**
    - Determines the persistence value for the angle effect, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle_lacunarity`**
    - Controls the lacunarity for the angle effect, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle_repeat`**
    - Sets the repeat interval for the angle effect in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trx_octaves`**
    - Specifies the number of octaves for the translation effect in the x-direction, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trx_persistence`**
    - Determines the persistence value for the translation effect in the x-direction, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trx_lacunarity`**
    - Controls the lacunarity for the translation effect in the x-direction, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trx_repeat`**
    - Sets the repeat interval for the translation effect in the x-direction in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`try_octaves`**
    - Specifies the number of octaves for the translation effect in the y-direction, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`try_persistence`**
    - Determines the persistence value for the translation effect in the y-direction, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`try_lacunarity`**
    - Controls the lacunarity for the translation effect in the y-direction, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`try_repeat`**
    - Sets the repeat interval for the translation effect in the y-direction in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trz_octaves`**
    - Specifies the number of octaves for the translation effect in the z-direction, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trz_persistence`**
    - Determines the persistence value for the translation effect in the z-direction, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trz_lacunarity`**
    - Controls the lacunarity for the translation effect in the z-direction, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trz_repeat`**
    - Sets the repeat interval for the translation effect in the z-direction in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotx_octaves`**
    - Specifies the number of octaves for the rotation effect around the x-axis, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotx_persistence`**
    - Determines the persistence value for the rotation effect around the x-axis, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotx_lacunarity`**
    - Controls the lacunarity for the rotation effect around the x-axis, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotx_repeat`**
    - Sets the repeat interval for the rotation effect around the x-axis in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roty_octaves`**
    - Specifies the number of octaves for the rotation effect around the y-axis, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roty_persistence`**
    - Determines the persistence value for the rotation effect around the y-axis, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`roty_lacunarity`**
    - Controls the lacunarity for the rotation effect around the y-axis, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`roty_repeat`**
    - Sets the repeat interval for the rotation effect around the y-axis in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotz_octaves`**
    - Specifies the number of octaves for the rotation effect around the z-axis, affecting the level of detail in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotz_persistence`**
    - Determines the persistence value for the rotation effect around the z-axis, influencing the amplitude of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotz_lacunarity`**
    - Controls the lacunarity for the rotation effect around the z-axis, which adjusts the frequency of each octave in the Perlin noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotz_repeat`**
    - Sets the repeat interval for the rotation effect around the z-axis in Perlin noise, defining how the noise pattern repeats over space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`opac_perlin_settings`**
    - Comfy dtype: `DICT`
    - Returns a configuration dictionary for Perlin noise settings, which can be applied to the OPAC node to influence animation keyframing.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPACPerlinSettings:
    """
        Configuration node for Perlin noise sampling in OPAC node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zoom_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "zoom_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "zoom_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "zoom_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "angle_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "angle_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "angle_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "angle_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "try_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "try_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "try_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "try_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "roty_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "roty_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "roty_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "roty_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("opac_perlin_settings",)
    FUNCTION = "process"
    CATEGORY = "SALT/Animation/Keyframing"

    def process(self, **kwargs):
        return (kwargs, )

```
