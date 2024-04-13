# Lora Syntax Processor (Mikey)
## Documentation
- Class name: `LoraSyntaxProcessor`
- Category: `Mikey/Lora`
- Output node: `False`

The LoraSyntaxProcessor node is designed to process text inputs by identifying and handling specific syntax related to LoRA (Low-Rank Adaptation) configurations. It dynamically loads and applies LoRA modifications to models and clips based on the directives found within the text, enhancing the flexibility and customization of model behavior.
## Input types
### Required
- **`model`**
    - The model parameter represents the initial model to which LoRA modifications will be applied. It plays a crucial role in the dynamic adaptation of the model's behavior based on the LoRA configurations specified in the text.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The clip parameter is the initial clip to which LoRA modifications will be applied, allowing for dynamic adjustments in clip behavior based on LoRA configurations found in the text.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.Tensor`
- **`text`**
    - The text parameter contains the input text which may include specific LoRA syntax that dictates how LoRA modifications should be applied to the model and clip.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - The seed parameter is used for processing random syntax within the text, ensuring reproducibility of the text processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model after applying LoRA modifications.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The modified clip after applying LoRA modifications.
    - Python dtype: `torch.Tensor`
- **`text`**
    - Comfy dtype: `STRING`
    - The processed text stripped of LoRA syntax.
    - Python dtype: `str`
- **`unprocessed_text`**
    - Comfy dtype: `STRING`
    - The original text before processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoraSyntaxProcessor:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "text": ("STRING", {"multiline": True, "default": "<lora:filename:weight>"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
                    },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def process(self, model, clip, text, seed, extra_pnginfo=None, prompt=None):
        # process random syntax
        text = process_random_syntax(text, seed)
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        clip_lora = clip
        if len(lora_prompts) > 0:
            # loop through each lora prompt
            for lora_prompt in lora_prompts:
                # get the lora filename
                lora_filename = lora_prompt[0]
                # check for file extension in filename
                if '.safetensors' not in lora_filename:
                    lora_filename += '.safetensors'
                # get the lora multiplier
                lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
                print('Loading LoRA: ' + lora_filename + ' with multiplier: ' + str(lora_multiplier))
                model, clip_lora = LoraLoader.load_lora(self, model, clip, lora_filename, lora_multiplier, lora_multiplier)
        # strip lora syntax from text
        stripped_text = re.sub(lora_re, '', stripped_text)
        return (model, clip_lora, stripped_text,  text, )

```
