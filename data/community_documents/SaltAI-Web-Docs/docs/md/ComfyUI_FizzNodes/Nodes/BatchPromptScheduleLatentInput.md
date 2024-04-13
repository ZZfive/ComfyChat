# Batch Prompt Schedule (Latent Input) 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleLatentInput`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed to handle the scheduling of latent inputs based on a batch prompt schedule. It allows for the dynamic adjustment of latent inputs over a sequence of frames or iterations, facilitating the creation of animations or variations in generated content.
## Input types
### Required
- **`text`**
    - The text parameter is crucial for defining the sequence of prompts or instructions that will guide the scheduling process. It affects the node's execution by determining the specific transformations or adjustments to be applied to the latent inputs over time.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter specifies the video clip or animation frame to which the latent inputs will be applied, playing a key role in aligning the scheduled changes with the visual content.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.Tensor`
- **`num_latents`**
    - This parameter indicates the number of latent inputs to be scheduled, directly influencing the complexity and variety of the generated content.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`print_output`**
    - A boolean flag that, when set to True, enables the printing of scheduling information for debugging or monitoring purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Optional pre-text that can be prepended to each prompt in the scheduling sequence, modifying the context or emphasis of the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional appended text that can be added to each prompt, further customizing the output based on the scheduling requirements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_frame`**
    - Defines the starting frame for the scheduling process, allowing for precise control over when the adjustments to latent inputs begin.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`pw_a`**
    - A weight parameter that influences the scheduling algorithm, potentially adjusting the rate or manner in which latent inputs are modified over time.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Similar to pw_a, this is another weight parameter that can affect the scheduling dynamics, offering additional customization of the content generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - This weight parameter allows for further fine-tuning of the scheduling mechanism, enabling more nuanced control over the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - The final weight parameter in the series, pw_d provides an additional layer of customization for the scheduling algorithm, impacting the overall output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The POS output represents the positive conditioning applied to the latent inputs as per the scheduling process, influencing the generation towards desired attributes.
    - Python dtype: `torch.Tensor`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The NEG output signifies the negative conditioning applied, guiding the generation away from certain attributes as dictated by the schedule.
    - Python dtype: `torch.Tensor`
- **`INPUT_LATENTS`**
    - Comfy dtype: `LATENT`
    - This output provides the adjusted latent inputs after being processed through the scheduling mechanism, ready for further use in content generation.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ConditioningCombine](../../Comfy/Nodes/ConditioningCombine.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - KSampler Adv. (Efficient)



## Source code
```python
class BatchPromptScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "num_latents": ("LATENT", ),
                             "print_output":("BOOLEAN", {"default": False}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "start_frame": ("INT", {"default": 0.0, "min": 0, "max": 9999, "step": 1, }),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", )
    RETURN_NAMES = ("POS", "NEG", "INPUT_LATENTS",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, print_output, clip, start_frame, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        max_frames = sum(tensor.size(0) for tensor in num_latents.values())
        max_frames += start_frame
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)

        animation_prompts = json.loads(inputText.strip())
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text,
                                                                           app_text, pw_a, pw_b, pw_c, pw_d,
                                                                           print_output)
        pc = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, )

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text,
                                                                           app_text, pw_a, pw_b, pw_c, pw_d,
                                                                           print_output)
        nc = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, )

        return (pc, nc, num_latents,)

```
