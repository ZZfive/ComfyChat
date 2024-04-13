# LLM Regex Creator
## Documentation
- Class name: `LLMRegexCreator`
- Category: `SALT/Llama-Index/Tools/Regex`
- Output node: `False`

The LLMRegexCreator node leverages a language model to generate regular expression patterns based on a given description. It aims to simplify the process of creating regex patterns by interpreting natural language inputs.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for generating the regex pattern. This model interprets the provided description and extra directions to create a well-formed regex.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `LLM_MODEL`
- **`description`**
    - A detailed description of the regex pattern to be created, which may optionally include examples. This serves as the primary input for the regex generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Additional instructions or constraints to guide the regex generation process, enhancing the specificity and accuracy of the resulting pattern.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`regex_pattern`**
    - Comfy dtype: `STRING`
    - The generated regex pattern that matches the criteria described in the input description.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMRegexCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Describe regex pattern to create, optionally provide example"}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("regex_pattern",)

    FUNCTION = "create_regex"
    CATEGORY = "SALT/Llama-Index/Tools/Regex"

    def create_regex(self, llm_model, description, extra_directions=""):
        prompt = (
            f"Create only a well formed regex pattern based on the following description:\n\n"
            f"{description}\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the regex pattern is concise and accurately matches the described criteria."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
