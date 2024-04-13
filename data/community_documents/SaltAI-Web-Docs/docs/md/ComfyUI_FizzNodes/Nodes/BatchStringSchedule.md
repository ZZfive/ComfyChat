# Batch String Schedule 📅🅕🅝
## Documentation
- Class name: `BatchStringSchedule`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

The BatchStringSchedule node is designed for scheduling and manipulating strings in batch operations. It focuses on processing multiple strings simultaneously, applying transformations, and generating scheduled outputs based on predefined criteria. This node is particularly useful in scenarios where batch processing of string data is required to achieve efficient and scalable text manipulation and generation.
## Input types
### Required
- **`text`**
    - The 'text' parameter is a multiline string that serves as the input for batch processing. It is crucial for defining the text data to be manipulated or transformed within the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - The 'max_frames' parameter specifies the maximum number of frames for the batch scheduling process. It determines the length of the animation or transformation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`pre_text`**
    - The 'pre_text' parameter allows for the specification of text to be prepended before the main text content in the scheduling process, influencing the overall output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - The 'app_text' parameter enables the appending of text after the main content, further customizing the output of the scheduling process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - The 'pw_a' parameter influences the weighting of the animation or transformation, contributing to the dynamic adjustment of the scheduled content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - The 'pw_b' parameter, similar to 'pw_a', affects the weighting and dynamic adjustment of the scheduled content in the batch processing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - The 'pw_c' parameter plays a role in the dynamic weighting and adjustment of the content, similar to 'pw_a' and 'pw_b', but with its unique influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - The 'pw_d' parameter, like the other weighting parameters, impacts the dynamic adjustment of the scheduled content, offering a distinct level of control.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - This output represents the processed text data after applying the scheduled transformations. It encapsulates the results of batch string scheduling and manipulation, formatted as a string.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchStringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),  # "forceInput": True}),
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

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        start_frame = 0
        animation_prompts = json.loads(inputText.strip())
        cur_prompt_series, nxt_prompt_series, weight_series = interpolate_prompt_series(animation_prompts, max_frames, start_frame, pre_text,
                                                             app_text, pw_a, pw_b, pw_c, pw_d)
        return (cur_prompt_series,)

```
