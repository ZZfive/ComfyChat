# String Schedule 📅🅕🅝
## Documentation
- Class name: `StringSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

The StringSchedule node is designed for scheduling string values over a specified number of frames, allowing for dynamic text manipulation based on frame count. It supports batch processing of key frames, enabling complex animation and timing sequences through text-based inputs.
## Input types
### Required
- **`text`**
    - The text input contains key frames for animation, allowing for dynamic changes in text based on the frame count. It supports multiline input for complex animations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Specifies the maximum number of frames for the animation, dictating the length and timing of the text-based animation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - Indicates the current frame number within the animation sequence, used to determine the specific text manipulation at that point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`pre_text`**
    - Optional pre-text that can be prepended to the animated text for each frame.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional appended text that can be added to the animated text for each frame.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - A weight parameter for adjusting the animation effect on the text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A weight parameter for adjusting the animation effect on the text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - A weight parameter for adjusting the animation effect on the text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - A weight parameter for adjusting the animation effect on the text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns the animated text value for the current frame, facilitating textual manipulations.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, })},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE",  }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        current_frame = current_frame % max_frames
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        cur_prompt = interpolate_string(animation_prompts, max_frames, current_frame, pre_text,
                                                             app_text, pw_a, pw_b, pw_c, pw_d)
        #c = PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (cur_prompt,)

```
