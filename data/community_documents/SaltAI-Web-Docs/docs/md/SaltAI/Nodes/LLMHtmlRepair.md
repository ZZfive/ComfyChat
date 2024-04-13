# LLM HTML Repair
## Documentation
- Class name: `LLMHtmlRepair`
- Category: `SALT/Llama-Index/Tools/HTML`
- Output node: `False`

The LLMHtmlRepair node leverages a language model to inspect and correct malformed HTML content, ensuring the output is valid HTML that retains the original data without alteration.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for repairing the HTML content. It's crucial for interpreting and correcting the malformed HTML.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
- **`text_input`**
    - The malformed HTML content that needs to be repaired. This input is essential for the node to understand what needs fixing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Additional instructions or guidelines for the language model to follow while repairing the HTML, allowing for more tailored repairs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`html_output`**
    - Comfy dtype: `STRING`
    - The repaired HTML content, which is well-structured, valid, and retains the original data.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMHtmlRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed HTML..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("html_output",)

    FUNCTION = "repair_html"
    CATEGORY = "SALT/Llama-Index/Tools/HTML"

    def repair_html(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed HTML, please inspect it and repair it so that it's valid HTML, without changing or losing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the HTML output is well-structured, valid,, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
