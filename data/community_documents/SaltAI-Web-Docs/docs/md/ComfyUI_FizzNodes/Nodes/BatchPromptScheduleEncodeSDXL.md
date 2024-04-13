# Batch Prompt Schedule SDXL 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleEncodeSDXL`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed to handle batch processing of prompt schedules specifically tailored for SDXL models. It efficiently manages the encoding of multiple prompts, applying transformations and scheduling techniques to optimize the generation process for large-scale or complex prompt sequences.
## Input types
### Required
- **`width`**
    - Specifies the width of the output image, directly influencing the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output image, affecting the overall size and aspect ratio of the generated visuals.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Specifies the width of the crop area, which is part of the image processing before encoding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Determines the height of the crop area, crucial for the image processing stage prior to encoding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width for the output image after processing, influencing the final dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for the output image after processing, affecting the final size and aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - The global input text containing prompts for the generation process, serving as a key content source.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter is essential for tokenizing and encoding the input text, serving as the foundation for generating the conditioned outputs.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`text_l`**
    - The local input text containing additional prompts, enriching the content generation with more specificity.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Defines the total number of frames to be processed, crucial for scheduling the prompt sequences over time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A flag indicating whether to print the output during processing, useful for debugging or monitoring.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text_G`**
    - Pre-text for global prompts, used to prepend text to global prompts for additional context or modification.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - Append text for global prompts, used to add text to the end of global prompts for further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Pre-text for local prompts, used to prepend text to local prompts for additional context or modification.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - Append text for local prompts, used to add text to the end of local prompts for further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A, part of the prompt weighting and scheduling mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, another component of the prompt weighting and scheduling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, contributing to the prompt weighting and scheduling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, also involved in the prompt weighting and scheduling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Returns the conditioning information, including the encoded prompts and associated metadata, ready for further processing or generation.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchPromptScheduleEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
            "print_output":("BOOLEAN", {"default": False}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G"}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G"}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L"}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L"}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, print_output, pw_a, pw_b, pw_c, pw_d):
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output,),)

```
