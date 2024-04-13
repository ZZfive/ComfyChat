# Parallax Motion Camera Scheduler
## Documentation
- Class name: `OPCScheduler`
- Category: `SALT/Animation/Keyframing/Parallax Motion`
- Output node: `False`

The OPCScheduler node is designed to generate a series of keyframe parameters for animation, specifically tailored to create parallax motion effects. It abstracts the complexity of calculating zoom, angle, and 3D translation and rotation values, facilitating the creation of dynamic and engaging visual content.
## Input types
### Required
- **`frame_count`**
    - Specifies the total number of frames for the animation, defining its duration.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`zoom_speed`**
    - Controls the speed of zooming in or out, affecting the pace at which the view is magnified or reduced.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pan_speed`**
    - Determines the speed of panning, influencing the rate of horizontal movement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pan_directions`**
    - Defines the directions for panning, contributing to the spatial orientation of the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`direction_change_frames`**
    - Specifies the frames at which the direction of panning changes, adding complexity to the motion.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`tremor_scale`**
    - Sets the scale for tremor effects, adding a subtle, dynamic shake to the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_octaves`**
    - Determines the number of layers for the tremor effect, influencing its complexity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tremor_persistence`**
    - Controls the persistence of the tremor effect across frames, affecting its continuity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_lacunarity`**
    - Sets the gap between successive tremor octaves, influencing the effect's granularity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`direction_curve`**
    - Defines the curve function for direction changes, affecting the smoothness of transitions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_x`**
    - Specifies the starting horizontal position, setting the initial point of view.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_y`**
    - Determines the starting vertical position, establishing the initial vertical orientation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_mode`**
    - Selects the mode of zooming (in, out, or both), defining the zoom behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`layer_offsets`**
    - Specifies the offsets for different layers, contributing to the depth of the parallax effect.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`float_layers`**
    - Comfy dtype: `LIST`
    - The calculated parameters for each layer in the animation, encapsulated as a series of floating-point values.
    - Python dtype: `List[List[float]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPCScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 60, "min": 1, "max": 4096}),
                "zoom_speed": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "pan_speed": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 5.0, "step": 0.001}),
                "pan_directions": ("STRING", {"default": "90,180,270"}),
                "direction_change_frames": ("STRING", {"default": "10,20,40"}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "direction_curve": (list(easing_functions.keys()), ),
                "start_x": ("FLOAT", {"default": 0, "min": -nodes.MAX_RESOLUTION, "max": nodes.MAX_RESOLUTION}),
                "start_y": ("FLOAT", {"default": 0}),
                "zoom_mode": (["zoom-in", "zoom-out", "zoom-in-out"], ),
                "layer_offsets": ("STRING", {"default": "1,0.8,0.6"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("float_layers",)
    FUNCTION = "execute"
    CATEGORY = "SALT/Animation/Keyframing/Parallax Motion"

    def __init__(self):
        self.scheduler = None

    def execute(self, **kwargs):

        self.process_kwargs(**kwargs)

        if not self.scheduler:
            raise Exception("Camera Scheduler was not initialized. Perhaps your settings are bugged?")
        
        mode = kwargs.get("zoom_mode", "zoom-in")
        layer_offsets = kwargs.get("layer_offsets", "1")

        animation_data = self.scheduler.animate(mode, layer_offsets)

        return (animation_data, )

    def process_kwargs(self, **kwargs):
        frame_count = kwargs.get("frame_count", 60)
        zoom_speed = kwargs.get("zoom_speed", 0.01)
        pan_speed = kwargs.get("pan_speed", 0.5)
        pan_directions = list(map(float, kwargs.get("pan_directions", "90,180,270").split(",")))
        direction_change_frames = list(map(int, kwargs.get("direction_change_frames", "10,20,40").split(",")))
        tremor_params = {
            "scale": kwargs.get("tremor_scale", 0.1),
            "octaves": kwargs.get("tremor_octaves", 1),
            "persistence": kwargs.get("tremor_persistence", 0.5),
            "lacunarity": kwargs.get("tremor_lacunarity", 2.0),
        }
        direction_curve = kwargs.get("direction_curve", "linear")
        start_x = kwargs.get("start_x", 0)
        start_y = kwargs.get("start_y", 0)

        self.scheduler = OrganicPerlinCameraScheduler(
            frame_count=frame_count,
            zoom_speed=zoom_speed,
            pan_speed=pan_speed,
            pan_directions=pan_directions,
            direction_change_frames=direction_change_frames,
            tremor_params=tremor_params,
            direction_curve=direction_curve,
            start_x=start_x,
            start_y=start_y,
        )

```
