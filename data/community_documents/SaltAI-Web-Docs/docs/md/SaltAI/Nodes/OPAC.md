# OPAC Scheduler
## Documentation
- Class name: `OPAC`
- Category: `SALT/Animation/Keyframing`
- Output node: `False`

The OPAC node is designed to schedule and manage animation keyframing tasks, providing a structured approach to applying object property animation changes over time. It serves as a central hub for initiating and controlling the animation process, ensuring that keyframes are applied accurately and efficiently.
## Input types
### Required
- **`use_wiggle`**
    - Determines whether to apply a 'wiggle' effect to the animation, adding a dynamic, non-linear motion to the object properties for a more natural and varied animation outcome.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frame_count`**
    - Specifies the total number of frames for the animation sequence, defining the duration and granularity of the keyframing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`zoom_range`**
    - Defines the range of zoom levels to be applied across the animation frames, allowing for dynamic focus and perspective changes.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`zoom_tremor_scale`**
    - Adjusts the intensity of the tremor effect applied to the zoom, enhancing the dynamic and varied animation outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle_range`**
    - Sets the range of angles for rotation, enabling diverse orientation changes throughout the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`angle_tremor_scale`**
    - Modifies the intensity of the tremor effect on the angle, contributing to a more lively and varied animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trx_range`**
    - Determines the range of translation along the x-axis, allowing for horizontal movement across frames.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`trx_tremor_scale`**
    - Controls the tremor effect's intensity on the x-axis translation, adding to the animation's dynamic nature.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`try_range`**
    - Specifies the range of translation along the y-axis, facilitating vertical movement throughout the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`try_tremor_scale`**
    - Adjusts the tremor effect's intensity on the y-axis translation, enhancing the animation's liveliness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trz_range`**
    - Sets the range of translation along the z-axis, enabling depth movement across the animation frames.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`trz_tremor_scale`**
    - Modifies the intensity of the tremor effect on the z-axis translation, contributing to a more dynamic animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotx_range`**
    - Defines the range of rotation around the x-axis, allowing for tilt movements throughout the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`rotx_tremor_scale`**
    - Controls the tremor effect's intensity on the x-axis rotation, adding to the animation's varied movements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`roty_range`**
    - Determines the range of rotation around the y-axis, enabling swivel movements across the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`roty_tremor_scale`**
    - Adjusts the tremor effect's intensity on the y-axis rotation, enhancing the animation's dynamic nature.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotz_range`**
    - Sets the range of rotation around the z-axis, allowing for spin movements throughout the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `tuple[float, float]`
- **`rotz_tremor_scale`**
    - Modifies the intensity of the tremor effect on the z-axis rotation, contributing to a more lively animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`opac_perlin_settings`**
    - Configures the Perlin noise settings for opacity variations, adding a layer of complexity and realism to the animation.
    - Comfy dtype: `DICT`
    - Python dtype: `dict`
## Output types
- **`zoom`**
    - Comfy dtype: `LIST`
    - The calculated zoom levels for each frame, providing a dynamic perspective change throughout the animation.
    - Python dtype: `list[float]`
- **`angle`**
    - Comfy dtype: `LIST`
    - The calculated angles for each frame, enabling diverse orientation changes throughout the animation.
    - Python dtype: `list[float]`
- **`translation_x`**
    - Comfy dtype: `LIST`
    - The calculated translations along the x-axis for each frame, allowing for horizontal movement across the animation.
    - Python dtype: `list[float]`
- **`translation_y`**
    - Comfy dtype: `LIST`
    - The calculated translations along the y-axis for each frame, facilitating vertical movement throughout the animation.
    - Python dtype: `list[float]`
- **`translation_z`**
    - Comfy dtype: `LIST`
    - The calculated translations along the z-axis for each frame, enabling depth movement across the animation frames.
    - Python dtype: `list[float]`
- **`rotation_3d_x`**
    - Comfy dtype: `LIST`
    - The calculated rotations around the x-axis for each frame, allowing for tilt movements throughout the animation.
    - Python dtype: `list[float]`
- **`rotation_3d_y`**
    - Comfy dtype: `LIST`
    - The calculated rotations around the y-axis for each frame, enabling swivel movements across the animation.
    - Python dtype: `list[float]`
- **`rotation_3d_z`**
    - Comfy dtype: `LIST`
    - The calculated rotations around the z-axis for each frame, allowing for spin movements throughout the animation.
    - Python dtype: `list[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPAC:
    """
        Generates semi-random keyframes for zoom, spin, and translation based on specified start and end ranges,
        with individual tremor scale controls for each parameter, allowing for organic variation using Perlin noise.
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_wiggle": ("BOOLEAN", {"default": True}),
                "frame_count": ("INT", {"default": 48, "min": 1, "max": 500}),
                "zoom_range": ("STRING", {"default": "0.95,1.05"}),
                "zoom_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "angle_range": ("STRING", {"default": "-5,5"}),
                "angle_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trx_range": ("STRING", {"default": "-10,10"}),
                "trx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "try_range": ("STRING", {"default": "-10,10"}),
                "try_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "trz_range": ("STRING", {"default": "-10,10"}),
                "trz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotx_range": ("STRING", {"default": "-5,5"}),
                "rotx_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "roty_range": ("STRING", {"default": "-5,5"}),
                "roty_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "rotz_range": ("STRING", {"default": "-5,5"}),
                "rotz_tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
            },
            "optional": {
                "opac_perlin_settings": ("DICT", {})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom", "angle", "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z")
    FUNCTION = "execute"
    CATEGORY = "SALT/Animation/Keyframing"

    def process_kwargs(self, **kwargs):
        self.use_wiggle = kwargs.get('use_wiggle', True)
        self.frame_count = kwargs.get('frame_count', 48)
        self.zoom_range = tuple(map(float, kwargs.get('zoom_range', "0.95,1.05").split(',')))
        self.zoom_tremor_scale = kwargs.get('zoom_tremor_scale', 0.05)
        self.angle_range = tuple(map(float, kwargs.get('angle_range', "-5,5").split(',')))
        self.angle_tremor_scale = kwargs.get('angle_tremor_scale', 0.05)
        self.trx_range = tuple(map(float, kwargs.get('trx_range', "-10,10").split(',')))
        self.trx_tremor_scale = kwargs.get('trx_tremor_scale', 0.05)
        self.try_range = tuple(map(float, kwargs.get('try_range', "-10,10").split(',')))
        self.try_tremor_scale = kwargs.get('try_tremor_scale', 0.05)
        self.trz_range = tuple(map(float, kwargs.get('trz_range', "-10,10").split(',')))
        self.trz_tremor_scale = kwargs.get('trz_tremor_scale', 0.05)
        self.rotx_range = tuple(map(float, kwargs.get('rotx_range', "-5,5").split(',')))
        self.rotx_tremor_scale = kwargs.get('rotx_tremor_scale', 0.05)
        self.roty_range = tuple(map(float, kwargs.get('roty_range', "-5,5").split(',')))
        self.roty_tremor_scale = kwargs.get('roty_tremor_scale', 0.05)
        self.rotz_range = tuple(map(float, kwargs.get('rotz_range', "-5,5").split(',')))
        self.rotz_tremor_scale = kwargs.get('rotz_tremor_scale', 0.05)

        # Zoom Perlin settings
        self.zoom_octaves = kwargs.get('zoom_octaves', 1)
        self.zoom_persistence = kwargs.get('zoom_persistence', 0.5)
        self.zoom_lacunarity = kwargs.get('zoom_lacunarity', 2.0)
        self.zoom_repeat = kwargs.get('zoom_repeat', 1024)
            
        # Angle Perlin settings
        self.angle_octaves = kwargs.get('angle_octaves', 1)
        self.angle_persistence = kwargs.get('angle_persistence', 0.5)
        self.angle_lacunarity = kwargs.get('angle_lacunarity', 2.0)
        self.angle_repeat = kwargs.get('angle_repeat', 1024)
            
        # Translation Perlin settings (trx, try, trz)
        self.trx_octaves = kwargs.get('trx_octaves', 1)
        self.trx_persistence = kwargs.get('trx_persistence', 0.5)
        self.trx_lacunarity = kwargs.get('trx_lacunarity', 2.0)
        self.trx_repeat = kwargs.get('trx_repeat', 1024)
            
        self.try_octaves = kwargs.get('try_octaves', 1)
        self.try_persistence = kwargs.get('try_persistence', 0.5)
        self.try_lacunarity = kwargs.get('try_lacunarity', 2.0)
        self.try_repeat = kwargs.get('try_repeat', 1024)
            
        self.trz_octaves = kwargs.get('trz_octaves', 1)
        self.trz_persistence = kwargs.get('trz_persistence', 0.5)
        self.trz_lacunarity = kwargs.get('trz_lacunarity', 2.0)
        self.trz_repeat = kwargs.get('trz_repeat', 1024)
            
        # Rotation Perlin settings (rotx, roty, rotz)
        self.rotx_octaves = kwargs.get('rotx_octaves', 1)
        self.rotx_persistence = kwargs.get('rotx_persistence', 0.5)
        self.rotx_lacunarity = kwargs.get('rotx_lacunarity', 2.0)
        self.rotx_repeat = kwargs.get('rotx_repeat', 1024)
        
        self.roty_octaves = kwargs.get('roty_octaves', 1)
        self.roty_persistence = kwargs.get('roty_persistence', 0.5)
        self.roty_lacunarity = kwargs.get('roty_lacunarity', 2.0)
        self.roty_repeat = kwargs.get('roty_repeat', 1024)
            
        self.rotz_octaves = kwargs.get('rotz_octaves', 1)
        self.rotz_persistence = kwargs.get('rotz_persistence', 0.5)
        self.rotz_lacunarity = kwargs.get('rotz_lacunarity', 2.0)
        self.rotz_repeat = kwargs.get('rotz_repeat', 1024)

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

    def execute(self, **kwargs):

        if kwargs.__contains__("opac_perlin_settings"):
            perlin_settings = kwargs.pop("opac_perlin_settings")
            kwargs.update(perlin_settings)
            print("\033[1m\033[94mOPAC Perlin Settings applied!:\033[0m")

        # Process the input values
        self.process_kwargs(**kwargs)

        if not self.use_wiggle:
            return ([0] * self.frame_count,) * 8

        # More dynamic implementation this time
        zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z = (
            [self.sample_perlin(i, *param) for i in range(self.frame_count)]
            for param in [
                (self.zoom_range[0], self.zoom_range[1], self.zoom_tremor_scale, self.zoom_octaves, self.zoom_persistence, self.zoom_lacunarity, self.zoom_repeat),
                (self.angle_range[0], self.angle_range[1], self.angle_tremor_scale, self.angle_octaves, self.angle_persistence, self.angle_lacunarity, self.angle_repeat),
                (self.trx_range[0], self.trx_range[1], self.trx_tremor_scale, self.trx_octaves, self.trx_persistence, self.trx_lacunarity, self.trx_repeat),
                (self.try_range[0], self.try_range[1], self.try_tremor_scale, self.try_octaves, self.try_persistence, self.try_lacunarity, self.try_repeat),
                (self.trz_range[0], self.trz_range[1], self.trz_tremor_scale, self.trz_octaves, self.trz_persistence, self.trz_lacunarity, self.trz_repeat),
                (self.rotx_range[0], self.rotx_range[1], self.rotx_tremor_scale, self.rotx_octaves, self.rotx_persistence, self.rotx_lacunarity, self.rotx_repeat),
                (self.roty_range[0], self.roty_range[1], self.roty_tremor_scale, self.roty_octaves, self.roty_persistence, self.roty_lacunarity, self.roty_repeat),
                (self.rotz_range[0], self.rotz_range[1], self.rotz_tremor_scale, self.rotz_octaves, self.rotz_persistence, self.rotz_lacunarity, self.rotz_repeat)
            ]
        )
            
        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")

        log_curve("zoom", zoom)
        log_curve("angle", angle)
        log_curve("translation_x", translation_x)
        log_curve("translation_y", translation_y)
        log_curve("translation_z", translation_z)
        log_curve("rotation_3d_x", rotation_3d_x)
        log_curve("rotation_3d_y", rotation_3d_y)
        log_curve("rotation_3d_z", rotation_3d_z)

        print("")

        return zoom, angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z

```
