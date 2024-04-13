# Prompt Schedule 📅🅕🅝
## Documentation
- Class name: `PromptSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

The PromptSchedule node is designed to manipulate and schedule text inputs for animation or sequential processing. It dynamically constructs JSON objects based on input text, frame numbers, and optional parameters, enabling the creation of scheduled or timed text sequences. This node is particularly useful in scenarios where text needs to be organized or presented in a time-based manner, such as in animations or scheduled displays.
## Input types
### Required
- **`text`**
    - The primary text input that will be scheduled or sequenced. It serves as the base content for the JSON object construction, playing a crucial role in the node's operation by determining the textual content of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A clip parameter that may be used in the animation or processing of the scheduled text.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
- **`max_frames`**
    - Defines the maximum number of frames for the animation or sequential process, setting an upper limit on the scheduling timeline.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - Specifies the current frame number in the sequence, affecting how the text is scheduled relative to the animation or process timeline.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag indicating whether to print the output for debugging or logging purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Optional text that can be prepended to the main text input, allowing for customization or addition of context before the scheduled text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional text that can be appended to the main text input, enabling further customization or additional context after the scheduled text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - A weight parameter for adjusting the influence of pre_text in the scheduling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A weight parameter for adjusting the influence of app_text in the scheduling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - An additional weight parameter for fine-tuning the scheduling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Another weight parameter for further fine-tuning the scheduling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - Returns the positive conditioning component of the scheduled text, as part of the animation or sequential processing.
    - Python dtype: `Conditioning`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - Returns the negative conditioning component of the scheduled text, contributing to the animation or sequential processing.
    - Python dtype: `Conditioning`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":defaultPrompt}),
            "clip": ("CLIP", ),
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
            "print_output":("BOOLEAN", {"default": False}),},# "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True,}),# "forceInput": True}),
            "app_text": ("STRING", {"multiline": True,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        current_frame = current_frame % max_frames
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        start_frame = 0
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        pc = PoolAnimConditioning(pos_cur_prompt[current_frame], pos_nxt_prompt[current_frame], weight[current_frame], clip)

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        nc = PoolAnimConditioning(neg_cur_prompt[current_frame], neg_nxt_prompt[current_frame], weight[current_frame], clip)

        return (pc, nc,)

```
