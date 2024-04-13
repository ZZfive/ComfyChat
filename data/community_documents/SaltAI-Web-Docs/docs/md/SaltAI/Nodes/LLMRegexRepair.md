# LLM Regex Repair
## Documentation
- Class name: `LLMRegexRepair`
- Category: `SALT/Llama-Index/Tools/Regex`
- Output node: `False`

This node leverages a large language model (LLM) to analyze and repair potentially malformed or incorrect regex patterns based on a given description and optional extra directions. It aims to ensure the regex pattern is well-formed and accurately matches the specified criteria.
## Input types
### Required
- **`llm_model`**
    - Specifies the large language model to use for repairing the regex pattern. It's crucial for interpreting the input pattern and description to generate a corrected version.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
- **`text_input`**
    - The potentially malformed or incorrect regex pattern that needs to be repaired.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`description`**
    - A description of what the regex pattern is intended to match, providing context for the repair process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Optional additional instructions or constraints to guide the LLM in repairing the regex pattern.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`repaired_regex_pattern`**
    - Comfy dtype: `STRING`
    - The repaired regex pattern that is well-formed and accurately matches the described criteria.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMRegexRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter the malformed regex pattern here"}),
                "description": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Describe what the regex pattern does wrong, and what it should do."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow, such as specific constraints or formats"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repaired_regex_pattern",)

    FUNCTION = "repair_regex"
    CATEGORY = "SALT/Llama-Index/Tools/Regex"

    def repair_regex(self, llm_model, text_input, description, extra_directions=""):
        prompt = (
            f"Given the potentially malformed or incorrect regex pattern:\n\n{text_input}\n\n"
            f"and the following description of what the pattern should match:\n\n{description}\n\n"
            f"{extra_directions}\n\n"
            "Please repair the regex pattern so it is well-formed and accurately matches the described criteria."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
