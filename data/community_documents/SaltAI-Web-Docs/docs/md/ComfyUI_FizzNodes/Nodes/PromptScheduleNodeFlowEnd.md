# Prompt Schedule NodeFlow End 📅🅕🅝
## Documentation
- Class name: `PromptScheduleNodeFlowEnd`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node is designed to process and refine animation prompts for a given frame within a scheduled animation flow. It evaluates and interpolates between positive and negative prompts based on the current frame, applying specified weights to adjust the influence of each prompt. The node aims to enhance the dynamic and temporal aspects of generated content by allowing for frame-specific prompt adjustments.
## Input types
### Required
- **`text`**
    - The JSON-formatted string containing the animation prompts. It serves as the basis for generating the frame-specific prompts by interpolating between the specified keyframes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter influences the conditioning of the animation, affecting how prompts are interpolated and weighted across frames.
    - Comfy dtype: `CLIP`
    - Python dtype: `float`
- **`max_frames`**
    - Specifies the total number of frames in the animation sequence. It is used to determine the current frame's position within the overall animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A flag to enable or disable printing of the interpolation process and results, aiding in debugging and visualization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`current_frame`**
    - Indicates the specific frame for which the prompts are being refined. This parameter is essential for calculating the interpolated prompts for the given frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`pre_text`**
    - Text to prepend to each prompt, used to modify or contextualize the animation prompts before interpolation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Text to append to each prompt, used to add additional context or modifications to the animation prompts after interpolation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A, used in the interpolation formula to adjust the influence of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, used in conjunction with other weight parameters to fine-tune the interpolation of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, plays a role in the interpolation process by adjusting the balance between different prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, used to further refine the weighting and interpolation of animation prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The processed and interpolated positive prompts for the specified frame, along with their corresponding weights.
    - Python dtype: `PoolAnimConditioning`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The processed and interpolated negative prompts for the specified frame, along with their corresponding weights.
    - Python dtype: `PoolAnimConditioning`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}), 
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
                            "print_output": ("BOOLEAN", {"default": False}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),}, #"forceInput": True}),},
               "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),#"forceInput": True}),
                            "app_text": ("STRING", {"multiline": True, "default": "APP", }),#"forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        current_frame = current_frame % max_frames
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        start_frame = 0
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        max_frames += start_frame
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        pc = PoolAnimConditioning(pos_cur_prompt[current_frame], pos_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        nc = PoolAnimConditioning(neg_cur_prompt[current_frame], neg_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        return (pc, nc,)

```
