# Value Schedule 📅🅕🅝
## Documentation
- Class name: `ValueSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

The ValueSchedule node is designed to animate values over a sequence of frames based on keyframes defined in a text input. It allows for dynamic value changes within animations, making it suitable for creating complex, time-based animations.
## Input types
### Required
- **`text`**
    - The text input containing keyframes for animation. It defines how values change over time, enabling the creation of dynamic animations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Specifies the maximum number of frames for the animation. It determines the animation's duration and how the keyframes are distributed across it.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - The current frame number in the animation sequence. It is used to calculate the specific value of the animation at this point in time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag that, when set to True, prints the current frame and its corresponding animated value. Useful for debugging or monitoring the animation progress.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - Returns the animated value at the current frame as a floating-point number.
    - Python dtype: `float`
- **`int`**
    - Comfy dtype: `INT`
    - Returns the animated value at the current frame, converted to an integer.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":defaultValue}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),# "forceInput": True}),
                             "print_output": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame, print_output):
        current_frame = current_frame % max_frames
        t = get_inbetweens(parse_key_frames(text, max_frames), max_frames)
        if (print_output is True):
            print("ValueSchedule: ",current_frame,"\n","current_frame: ",current_frame)
        return (t[current_frame],int(t[current_frame]),)

```
