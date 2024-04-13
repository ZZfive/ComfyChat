# Prompt Schedule NodeFlow 📅🅕🅝
## Documentation
- Class name: `PromptScheduleNodeFlow`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node is designed to schedule and sequence text inputs for generating or manipulating strings in a structured format. It dynamically constructs JSON objects based on the input text, number of frames, and optional parameters, facilitating the creation of scheduled text sequences for various applications.
## Input types
### Required
- **`text`**
    - The primary text input that will be included in the generated JSON object. It serves as the base content for the scheduling operation. The content of this text directly influences the content of the output JSON, acting as the main subject of the scheduling.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`num_frames`**
    - Specifies the number of frames to be added to the maximum frames, influencing the final value in the generated JSON object. This parameter directly affects the calculation of the new maximum frame count, which is critical for determining the duration or extent of the scheduling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`in_text`**
    - An optional initial text input that can be prepended to the generated JSON object, allowing for the extension or modification of existing sequences. This input modifies the starting point of the JSON object, effectively altering the initial state of the scheduled sequence.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Defines the maximum number of frames to be considered in the scheduling operation, affecting the construction of the JSON object. This parameter sets the upper limit for the frame count, impacting how the final scheduled sequence is structured.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The new maximum frame count after adding num_frames to max_frames. This output is crucial for understanding the new scope of the scheduling operation.
    - Python dtype: `int`
- **`string`**
    - Comfy dtype: `STRING`
    - The constructed JSON object as a string, representing the scheduled text sequence. This output encapsulates the result of the scheduling, providing a structured representation of the scheduled content.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptScheduleNodeFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),                           
                            "num_frames": ("INT", {"default": 24.0, "min": 0.0, "max": 9999.0, "step": 1.0}),},
               "optional":  {"in_text": ("STRING", {"multiline": False, }), # "forceInput": True}),
                             "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,})}} # "forceInput": True}),}}
    
    RETURN_TYPES = ("INT","STRING",)
    FUNCTION = "addString"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def addString(self, text, in_text='', max_frames=0, num_frames=0):
        if in_text:
            # Remove trailing comma from in_text if it exists
            in_text = in_text.rstrip(',')

        new_max = num_frames + max_frames

        if max_frames == 0:
            # Construct a new JSON object with a single key-value pair
            new_text = in_text + (', ' if in_text else '') + f'"{max_frames}": "{text}"'
        else:
            # Construct a new JSON object with a single key-value pair
            new_text = in_text + (', ' if in_text else '') + f'"{new_max}": "{text}"'



        return (new_max, new_text,)

```
