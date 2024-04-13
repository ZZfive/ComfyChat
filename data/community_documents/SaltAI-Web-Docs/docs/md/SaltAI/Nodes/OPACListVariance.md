# Apply Keyframing to List
## Documentation
- Class name: `OPACListVariance`
- Category: `SALT/Animation/Keyframing`
- Output node: `False`

The OPACListVariance node is designed to apply Perlin noise-based variance to a list of values, creating a sequence that simulates natural, organic fluctuations. This process enhances the dynamic quality of animations by introducing subtle variations in keyframe values, making the motion appear more lifelike and less mechanical.
## Input types
### Required
- **`list_input`**
    - The primary list of values to which the Perlin noise-based variance will be applied. This list represents the keyframe values that will be enhanced with natural fluctuations.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
### Optional
- **`tremor_scale`**
    - Determines the intensity of the Perlin noise-induced fluctuations applied to the list values. A higher tremor scale results in more pronounced variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`octaves`**
    - Controls the level of detail in the Perlin noise generation, affecting the complexity of the resulting fluctuations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`persistence`**
    - Influences the amplitude of each octave in the Perlin noise, affecting the smoothness of the fluctuations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lacunarity`**
    - Affects the frequency of each octave in the Perlin noise, influencing the granularity of the fluctuations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`repeat`**
    - Specifies the repetition rate of the Perlin noise pattern, ensuring a consistent pattern of fluctuations across multiple uses.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`list`**
    - Comfy dtype: `LIST`
    - The modified list of values, now enhanced with Perlin noise-based variance to simulate natural fluctuations.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPACListVariance:
    """
        Applies Perlin noise to each value in a list to create a OPAC Schedule out of it
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
            "optional": {
                "tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}), 
                "octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "opac_variance"
    CATEGORY = "SALT/Animation/Keyframing"

    #def sample_perlin(self, base, scale, x, min_val, max_val, octaves=1, persistence=0.5, lacunarity=2.0, repeat=1024):
    #    noise_val = self.perlin_noise.sample(base + x * scale, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    #    return noise_val * (max_val - min_val) + min_val
    
    def sample_perlin(self, frame_index, range_min, range_max, tremor_scale, octaves, persistence, lacunarity, repeat):
        # Prepare noise correctly with normalization
        t = frame_index / (self.frame_count - 1 if self.frame_count > 1 else 1)
        linear_value = (range_max - range_min) * t + range_min
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        interpolated_value = linear_value * noise_adjustment
        return interpolated_value

    def opac_variance(self, list_input, tremor_scale, octaves, persistence, lacunarity, repeat):
        self.frame_count = len(list_input) 
        varied_list = [
            self.sample_perlin(i, min(list_input), max(list_input), tremor_scale, octaves, persistence, lacunarity, self.frame_count)
            for i, _ in enumerate(self.frame_count)
        ]

        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("List Curve", varied_list)

        return (varied_list,)

```
