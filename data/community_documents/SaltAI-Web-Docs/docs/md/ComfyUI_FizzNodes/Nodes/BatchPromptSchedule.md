# Batch Prompt Schedule 📅🅕🅝
## Documentation
- Class name: `BatchPromptSchedule`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

The BatchPromptSchedule node is designed to process animation prompts in batches, evaluating and interpolating them based on specified parameters such as pre-text, appended text, and weights. It aims to facilitate the generation of dynamic, frame-specific text prompts for animations, enhancing the creative process by allowing for nuanced control over the narrative flow and thematic elements across different frames.
## Input types
### Required
- **`text`**
    - The animation prompt text, which serves as the base content for interpolation and evaluation across frames. It plays a crucial role in defining the narrative or thematic elements to be manipulated over the animation sequence.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter is used to specify the clipping range or limits for the animation, ensuring that the generated content remains within desired visual or thematic boundaries.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
- **`max_frames`**
    - Specifies the total number of frames for the animation, setting the boundary for how the prompts are interpolated and evaluated. It determines the length of the animation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag indicating whether to print the output of the interpolation and evaluation process. This can be useful for debugging or for providing insights into the generated content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Text to be prepended to the animation prompt for the current frame. This allows for additional contextual or thematic elements to be introduced at the beginning of the prompt.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Text to be appended to the animation prompt for the current frame. This enables the addition of concluding elements or further narrative details at the end of the prompt.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_frame`**
    - Indicates the starting frame from which the animation sequence begins. This parameter is essential for determining the initial point of interpolation and evaluation within the overall narrative flow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`pw_a`**
    - Weight parameter A, influencing the interpolation and evaluation of the animation prompts. It affects the balance and transition between different narrative or thematic elements across frames.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, further influencing the interpolation and evaluation process. It works in conjunction with other weight parameters to fine-tune the dynamic content generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, contributing to the nuanced control over the interpolation and evaluation of prompts. It plays a role in adjusting the narrative flow and thematic emphasis.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, affecting the interpolation and evaluation of animation prompts. It helps in achieving a desired balance and transition in the narrative or thematic content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The processed conditioning for the positive prompts, reflecting the interpolated and evaluated content for the current frame based on the specified parameters.
    - Python dtype: `object`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The processed conditioning for the negative prompts, mirroring the treatment of positive prompts but for contrasting or opposing narrative elements.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [Prompts Everywhere](../../cg-use-everywhere/Nodes/Prompts Everywhere.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - LinearBatchCreativeInterpolation



## Source code
```python
class BatchPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "print_output":("BOOLEAN", {"default": False}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE" }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP" }),  # "forceInput": True}),
                             "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, }),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, print_output, clip, start_frame, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        max_frames += start_frame
        animation_prompts = json.loads(inputText.strip())
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, print_output)
        pc = BatchPoolAnimConditioning( pos_cur_prompt, pos_nxt_prompt, weight, clip,)

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, print_output)
        nc = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, )

        return (pc, nc, )

```
