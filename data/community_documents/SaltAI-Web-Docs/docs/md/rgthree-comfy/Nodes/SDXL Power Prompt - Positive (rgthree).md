# SDXL Power Prompt - Positive (rgthree)
## Documentation
- Class name: `SDXL Power Prompt - Positive (rgthree)`
- Category: `rgthree`
- Output node: `False`

This node specializes in generating positive conditioning for prompts, leveraging the power of SDXL and CLIP models. It is designed to enhance the quality and relevance of generated images by applying positive conditioning techniques.
## Input types
### Required
- **`prompt_g`**
    - The global prompt input, which provides a broad context or theme for the image generation process. It's crucial for setting the overall direction and tone of the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt_l`**
    - The local prompt input, offering more specific guidance or details to fine-tune the image generation. It complements the global prompt by adding depth and specificity to the desired output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`opt_model`**
    - An optional model input that allows for further customization and refinement of the generated images.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`opt_clip`**
    - An optional CLIP model input to further refine the conditioning based on visual concepts, enhancing the alignment between text prompts and generated images.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`opt_clip_width`**
    - Specifies the width for the CLIP model's input, influencing the aspect ratio of the conditioning and, consequently, the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`opt_clip_height`**
    - Defines the height for the CLIP model's input, affecting the aspect ratio of the conditioning and the resulting images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`insert_lora`**
    - Allows the insertion of a LoRA model to modify the behavior of the underlying model dynamically.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`insert_embedding`**
    - Enables the insertion of a specific embedding to influence the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`insert_saved`**
    - Permits the use of a previously saved prompt or configuration, facilitating reuse and consistency in generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`target_width`**
    - The target width for the generated image, which can override the default width derived from the CLIP model's input if specified.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for the generated image, allowing to override the default height from the CLIP model's input if provided.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_width`**
    - Determines the width of the crop applied to the CLIP model's input, which can influence the focus area of the conditioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_height`**
    - Specifies the height of the crop applied to the CLIP model's input, potentially affecting the focus area of the conditioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`CONDITIONING`**
    - Comfy dtype: `CONDITIONING`
    - unknown
    - Python dtype: `unknown`
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
- **`CLIP`**
    - Comfy dtype: `CLIP`
    - unknown
    - Python dtype: `unknown`
- **`TEXT_G`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
- **`TEXT_L`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RgthreeSDXLPowerPromptPositive:
  """The Power Prompt for positive conditioning."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    SAVED_PROMPTS_FILES = folder_paths.get_filename_list('saved_prompts')
    SAVED_PROMPTS_CONTENT = []
    for filename in SAVED_PROMPTS_FILES:
      with open(folder_paths.get_full_path('saved_prompts', filename), 'r') as f:
        SAVED_PROMPTS_CONTENT.append(f.read())
    return {
      'required': {
        'prompt_g': ('STRING', {
          'multiline': True
        }),
        'prompt_l': ('STRING', {
          'multiline': True
        }),
      },
      'optional': {
        "opt_model": ("MODEL",),
        "opt_clip": ("CLIP",),
        "opt_clip_width": ("INT", {
          "forceInput": True,
          "default": 1024.0,
          "min": 0,
          "max": MAX_RESOLUTION
        }),
        "opt_clip_height": ("INT", {
          "forceInput": True,
          "default": 1024.0,
          "min": 0,
          "max": MAX_RESOLUTION
        }),
        'insert_lora': (['CHOOSE', 'DISABLE LORAS'] +
                        [os.path.splitext(x)[0] for x in folder_paths.get_filename_list('loras')],),
        'insert_embedding': ([
          'CHOOSE',
        ] + [os.path.splitext(x)[0] for x in folder_paths.get_filename_list('embeddings')],),
        'insert_saved': ([
          'CHOOSE',
        ] + SAVED_PROMPTS_FILES,),
        # We'll hide these in the UI for now.
        "target_width": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "target_height": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "crop_width": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
        "crop_height": ("INT", {
          "default": -1,
          "min": -1,
          "max": MAX_RESOLUTION
        }),
      },
      'hidden': {
        'values_insert_saved': (['CHOOSE'] + SAVED_PROMPTS_CONTENT,),
      }
    }

  RETURN_TYPES = ('CONDITIONING', 'MODEL', 'CLIP', 'STRING', 'STRING')
  RETURN_NAMES = ('CONDITIONING', 'MODEL', 'CLIP', 'TEXT_G', 'TEXT_L')
  FUNCTION = 'main'

  def main(self,
           prompt_g,
           prompt_l,
           opt_model=None,
           opt_clip=None,
           opt_clip_width=None,
           opt_clip_height=None,
           insert_lora=None,
           insert_embedding=None,
           insert_saved=None,
           target_width=-1,
           target_height=-1,
           crop_width=-1,
           crop_height=-1,
           values_insert_saved=None):

    if insert_lora == 'DISABLE LORAS':
      prompt_g, loras_g = get_and_strip_loras(prompt_g, True)
      prompt_l, loras_l = get_and_strip_loras(prompt_l, True)
      loras = loras_g + loras_l
      log_node_info(
        NODE_NAME,
        f'Disabling all found loras ({len(loras)}) and stripping lora tags for TEXT output.')
    elif opt_model != None and opt_clip != None:
      prompt_g, loras_g = get_and_strip_loras(prompt_g)
      prompt_l, loras_l = get_and_strip_loras(prompt_l)
      loras = loras_g + loras_l
      if len(loras):
        for lora in loras:
          opt_model, opt_clip = LoraLoader().load_lora(opt_model, opt_clip, lora['lora'],
                                                       lora['strength'], lora['strength'])
          log_node_success(NODE_NAME, f'Loaded "{lora["lora"]}" from prompt')
        log_node_info(NODE_NAME, f'{len(loras)} Loras processed; stripping tags for TEXT output.')
    elif '<lora:' in prompt_g or '<lora:' in prompt_l:
      _prompt_stripped_g, loras_g = get_and_strip_loras(prompt_g, True)
      _prompt_stripped_l, loras_l = get_and_strip_loras(prompt_l, True)
      loras = loras_g + loras_l
      if len(loras):
        log_node_warn(
          NODE_NAME, f'Found {len(loras)} lora tags in prompt but model & clip were not supplied!')
        log_node_info(NODE_NAME, 'Loras not processed, keeping for TEXT output.')

    conditioning = self.get_conditioning(prompt_g, prompt_l, opt_clip, opt_clip_width,
                                         opt_clip_height, target_width, target_height, crop_width,
                                         crop_height)

    return (conditioning, opt_model, opt_clip, prompt_g, prompt_l)

  def get_conditioning(self, prompt_g, prompt_l, opt_clip, opt_clip_width, opt_clip_height,
                       target_width, target_height, crop_width, crop_height):
    """Checks the inputs and gets the conditioning."""
    conditioning = None
    if opt_clip is not None:
      if opt_clip_width and opt_clip_height:
        target_width = target_width if target_width and target_width > 0 else opt_clip_width
        target_height = target_height if target_height and target_height > 0 else opt_clip_height
        crop_width = crop_width if crop_width and crop_width > 0 else 0
        crop_height = crop_height if crop_height and crop_height > 0 else 0
        conditioning = CLIPTextEncodeSDXL().encode(opt_clip, opt_clip_width, opt_clip_height,
                                                   crop_width, crop_height, target_width,
                                                   target_height, prompt_g, prompt_l)[0]
      else:
        # If we got an opt_clip, but no clip_width or _height, then use normal CLIPTextEncode
        log_node_info(
          self.NAME,
          'CLIP supplied, but not CLIP_WIDTH and CLIP_HEIGHT. Text encoding will use standard encoding with prompt_g and prompt_l concatenated.'
        )
        conditioning = CLIPTextEncode().encode(
          opt_clip, f'{prompt_g if prompt_g else ""}\n{prompt_l if prompt_l else ""}')[0]
    return conditioning

```
