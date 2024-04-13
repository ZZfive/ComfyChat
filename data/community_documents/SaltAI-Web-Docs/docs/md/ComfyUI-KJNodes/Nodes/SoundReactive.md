# SoundReactive
## Documentation
- Class name: `SoundReactive`
- Category: `KJNodes/experimental`
- Output node: `False`

The SoundReactive node is designed to process audio input by adjusting the sound level based on a range of frequencies, a multiplier, and a smoothing factor. It optionally normalizes the sound level, providing a flexible tool for audio-reactive applications.
## Input types
### Required
- **`sound_level`**
    - Represents the initial sound level to be adjusted. It's a crucial parameter as it directly influences the output sound level after processing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_range_hz`**
    - Defines the lower bound of the frequency range of interest. It helps in focusing the sound level adjustment within a specific frequency spectrum.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_range_hz`**
    - Sets the upper limit of the frequency range, allowing the node to target adjustments within a defined spectral range.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Acts as a scaling factor for the sound level, enabling amplification or attenuation of the input sound level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`smoothing_factor`**
    - Determines the degree of smoothing applied to the sound level, affecting the output's smoothness and stability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`normalize`**
    - A boolean flag that, when true, normalizes the sound level to a standard range, enhancing consistency across different inputs.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`sound_level`**
    - Comfy dtype: `FLOAT`
    - The adjusted sound level after processing, reflecting the effects of the input parameters.
    - Python dtype: `float`
- **`sound_level_int`**
    - Comfy dtype: `INT`
    - An integer representation of the adjusted sound level, providing a discrete output option.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SoundReactive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "sound_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 99999, "step": 0.01}),
            "start_range_hz": ("INT", {"default": 150, "min": 0, "max": 9999, "step": 1}),
            "end_range_hz": ("INT", {"default": 2000, "min": 0, "max": 9999, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99999, "step": 0.01}),
            "smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            }
    
    RETURN_TYPES = ("FLOAT","INT",)
    RETURN_NAMES =("sound_level", "sound_level_int",)
    FUNCTION = "react"

    CATEGORY = "KJNodes/experimental"
        
    def react(self, sound_level, start_range_hz, end_range_hz, smoothing_factor, multiplier, normalize):

        sound_level *= multiplier

        if normalize:
            sound_level /= 255

        sound_level_int = int(sound_level)
        return (sound_level, sound_level_int, )     

```
