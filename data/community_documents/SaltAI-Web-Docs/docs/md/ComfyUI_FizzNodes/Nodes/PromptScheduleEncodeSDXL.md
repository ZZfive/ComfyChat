# Prompt Schedule SDXL 📅🅕🅝
## Documentation
- Class name: `PromptScheduleEncodeSDXL`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node specializes in the dynamic scheduling of prompts for the SDXL model, enabling the interpolation and scheduling of text prompts over a sequence of frames. It evaluates expressions within prompts, sequences current and next prompts along with their conditioning strengths, and returns the conditioning for the current frame.
## Input types
### Required
- **`width`**
    - Specifies the width of the input image, affecting the conditioning's spatial dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the input image, influencing the spatial dimensions of the conditioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Specifies the width of the crop area, affecting the conditioning's spatial dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Determines the height of the crop area, influencing the spatial dimensions of the conditioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width for resizing the generated images, impacting the final output dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for resizing the generated images, affecting the final output dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - Contains the global prompts used for generating the conditioning across all frames.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter represents the model used for tokenizing and encoding the text prompts. It plays a crucial role in generating the conditioning for each frame based on the provided prompts.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`text_l`**
    - Contains the local prompts used for generating the conditioning specific to each frame.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - The total number of frames for which the prompts are scheduled, defining the sequence length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - The current frame number for which the conditioning is being generated, determining the specific prompt to use.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A flag indicating whether to print the output for debugging purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text_G`**
    - Prepended text to the global prompts, modifying the initial context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - Appended text to the global prompts, altering the final context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Prepended text to the local prompts, modifying the initial context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - Appended text to the local prompts, altering the final context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A, influencing the interpolation of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, affecting the interpolation of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, impacting the interpolation of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, determining the interpolation of prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Returns the conditioning for the current frame, which includes the encoded prompts along with additional metadata such as aesthetic score and image dimensions.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Union[float, int]]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PromptScheduleEncodeSDXL:
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
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0}),
            "print_output":("BOOLEAN", {"default": False})},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G",}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G",}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L",}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L",}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, current_frame, print_output, pw_a, pw_b, pw_c, pw_d):
        current_frame = current_frame % max_frames
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (interpolate_prompts_SDXL(animation_promptsG, animation_promptsL, max_frames, current_frame, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output,),)

```
