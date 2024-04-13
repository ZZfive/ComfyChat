# Batch Prompt Schedule SDXL (Latent Input) 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleSDXLLatentInput`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed for generating a sequence of prompts with specific weights and adjustments for animation purposes, tailored for use with latent inputs in the context of SDXL models. It processes a batch of animation prompts, applying transformations based on provided parameters to create a nuanced animation sequence that integrates smoothly with latent input data.
## Input types
### Required
- **`width`**
    - Specifies the width of the output image or animation frame. It's essential for defining the spatial dimensions of the visual content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the output image or animation frame. Along with width, it determines the spatial dimensions of the visual content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Defines the width of the crop area, allowing for focused animation on a specific portion of the image or frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Defines the height of the crop area, enabling focused animation on a particular segment of the image or frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width for resizing the animation frames, which can be different from the original width to fit specific requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for resizing the animation frames, which can be different from the original height to meet specific needs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - The global text parameter containing animation prompts and their respective frame numbers for the entire animation sequence. It serves as a blueprint for the global narrative and visual progression.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A parameter representing the clip model used for generating the animation. It plays a crucial role in the visual style and quality of the output.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`text_l`**
    - The local text parameter containing animation prompts and their respective frame numbers for specific segments of the animation. It complements the global text in defining the narrative and visual details.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`num_latents`**
    - Specifies the number of latent vectors used in the animation, which influences the complexity and variety of the visual content.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`print_output`**
    - A boolean flag indicating whether to print the output for debugging purposes. It helps in monitoring the animation generation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text_G`**
    - Global text to prepend to each prompt in the animation sequence, allowing for consistent thematic or narrative elements to be introduced across all frames globally.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - Global text to append to each prompt in the animation sequence, facilitating the inclusion of consistent narrative or thematic elements throughout the animation on a global scale.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Local text to prepend to each prompt in specific segments of the animation, allowing for focused thematic or narrative elements in those areas.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - Local text to append to each prompt in specific segments of the animation, facilitating the inclusion of focused narrative or thematic elements in those areas.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter 'a' for adjusting the influence of the current prompt in the animation sequence. It allows for fine-tuning the blending and transition between prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter 'b' for adjusting the influence of the next prompt in the animation sequence. It enables precise control over the narrative flow and visual transitions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter 'c' for modifying the overall aesthetic and thematic elements of the animation. It contributes to the unique visual style and mood of the sequence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter 'd' for further refining the animation's visual and thematic consistency. It complements the other weight parameters in achieving a cohesive animation experience.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output includes the processed animation prompts with specific weights and adjustments, tailored for integration with latent input data in SDXL models. It provides the necessary information for generating nuanced animations.
    - Python dtype: `Dict[str, Any]`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Outputs the latent vectors that have been adjusted and prepared for generating the animation sequence. These vectors are crucial for the visual content generation in SDXL models.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchPromptScheduleEncodeSDXLLatentInput:
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
            "num_latents": ("LATENT", ),
            "print_output":("BOOLEAN", {"default": False}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G",}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G",}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L",}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L",}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, num_latents, print_output, pw_a, pw_b, pw_c, pw_d):
        max_frames = sum(tensor.size(0) for tensor in num_latents.values())
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output, ), num_latents, )

```
