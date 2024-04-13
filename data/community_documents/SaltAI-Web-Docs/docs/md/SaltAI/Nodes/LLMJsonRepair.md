# LLM JSON Repair
## Documentation
- Class name: `LLMJsonRepair`
- Category: `SALT/Llama-Index/Tools/JSON`
- Output node: `False`

The LLMJsonRepair node is designed to inspect and repair malformed JSON strings, ensuring they are valid and well-formatted without losing any original data. It leverages a language model to understand and correct the structure and syntax of the input JSON.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for repairing the malformed JSON. It plays a crucial role in understanding and correcting the JSON structure and syntax.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
- **`text_input`**
    - The malformed JSON string that needs to be repaired. This input is critical as it provides the actual content that the node will attempt to correct.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Additional instructions or guidelines for the language model to follow while repairing the JSON. This can help tailor the repair process to specific requirements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The repaired and well-formatted JSON string, ensuring that it is valid and retains all original data.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMJsonRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Malformed JSON..."}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "compose_json"
    CATEGORY = "SALT/Llama-Index/Tools/JSON"

    def compose_json(self, llm_model, text_input, extra_directions=""):
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above malformed JSON, please inspect it and repair it so that it's valid JSON, without changing or loosing any data if possible."
            f"{extra_directions}\n\n"
            "Please ensure the JSON output is properly formatted, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
