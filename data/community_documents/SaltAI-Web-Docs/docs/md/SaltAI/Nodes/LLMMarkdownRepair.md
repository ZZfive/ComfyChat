# LLM Markdown Repair
## Documentation
- Class name: `LLMMarkdownRepair`
- Category: `SALT/Llama-Index/Tools/Markdown`
- Output node: `False`

The LLMMarkdownRepair node is designed to repair malformed Markdown text. It leverages a language model to analyze and correct the given Markdown, ensuring the output is valid, well-structured, and retains all original data.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for repairing the Markdown. It's crucial for interpreting and correcting the malformed input.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
- **`text_input`**
    - The malformed Markdown text that needs to be repaired. This is the primary data the node works on.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Optional additional instructions for the language model, guiding the repair process more specifically.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`markdown_output`**
    - Comfy dtype: `STRING`
    - The repaired Markdown text, which is valid, well-structured, and does not omit any original data.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMMarkdownRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed Markdown..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown_output",)

    FUNCTION = "repair_markdown"
    CATEGORY = "SALT/Llama-Index/Tools/Markdown"

    def repair_markdown(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed Markdown, please inspect it and repair it so that it's valid Markdown, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
