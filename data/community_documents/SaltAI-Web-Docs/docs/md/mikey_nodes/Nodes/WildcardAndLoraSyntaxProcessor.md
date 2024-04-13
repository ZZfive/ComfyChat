# Wildcard And Lora Syntax Processor (Mikey)
## Documentation
- Class name: `WildcardAndLoraSyntaxProcessor`
- Category: `Mikey/Lora`
- Output node: `False`

The WildcardAndLoraSyntaxProcessor node is designed to process text inputs by identifying and handling both wildcard and LoRA (Low-Rank Adaptation) syntaxes. It aims to enhance the flexibility and expressiveness of text inputs by allowing users to incorporate dynamic content through wildcards and to modify model behavior with LoRA specifications.
## Input types
### Required
- **`model`**
    - The 'model' parameter represents the model that will be potentially modified by the LoRA specifications contained within the text input.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The 'clip' parameter is used alongside the model and may be modified based on LoRA specifications found in the text input.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`text`**
    - The 'text' parameter is the primary text input that may contain wildcard and LoRA syntaxes. It serves as the basis for dynamic content generation and model behavior modification.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - The 'seed' parameter is used to ensure reproducibility in the generation of dynamic content from wildcards. It influences the randomness of content generated from wildcard expressions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model after processing LoRA specifications.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The potentially modified clip after processing LoRA specifications.
    - Python dtype: `torch.nn.Module`
- **`text`**
    - Comfy dtype: `STRING`
    - The processed text where wildcard and LoRA syntaxes have been resolved, resulting in dynamic content generation.
    - Python dtype: `str`
- **`unprocessed_text`**
    - Comfy dtype: `STRING`
    - The original text input before processing, which may contain unresolved wildcard and LoRA syntaxes.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WildcardAndLoraSyntaxProcessor:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "text": ("STRING", {"multiline": True, "default": "<lora:filename:weight>"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def extract_and_load_loras(self, text, model, clip):
        # load loras detected in the prompt text
        # The text for adding LoRA to the prompt, <lora:filename:multiplier>, is only used to enable LoRA, and is erased from prompt afterwards
        # The multiplier is optional, and defaults to 1.0
        # We update the model and clip, and return the new model and clip with the lora prompt stripped from the text
        # If multiple lora prompts are detected we chain them together like: original clip > clip_with_lora1 > clip_with_lora2 > clip_with_lora3 > etc
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
                # apply the lora to the clip using the LoraLoader.load_lora function
                # def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
                # ...
                # return (model_lora, clip_lora)
                # apply the lora to the clip
                model, clip_lora = LoraLoader.load_lora(self, model, clip, lora_filename, lora_multiplier, lora_multiplier)
        # strip lora syntax from text
        stripped_text = re.sub(lora_re, '', stripped_text)
        return model, clip, stripped_text

    def process(self, model, clip, text, seed, extra_pnginfo=None, prompt=None):
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        # process random syntax
        text = process_random_syntax(text, seed)
        # process wildcards
        text_ = find_and_replace_wildcards(text, seed, True)
        if len(text_) != len(text):
            seed = random.randint(0, 1000000)
        else:
            seed = 0
        # extract and load loras
        model, clip, stripped_text = self.extract_and_load_loras(text_, model, clip)
        # process wildcards again
        stripped_text = find_and_replace_wildcards(stripped_text, seed, True)
        return (model, clip, stripped_text, text_, )

```
