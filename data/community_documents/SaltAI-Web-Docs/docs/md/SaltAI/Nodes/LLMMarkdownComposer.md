# LLM Markdown Composer
## Documentation
- Class name: `LLMMarkdownComposer`
- Category: `SALT/Llama-Index/Tools/Markdown`
- Output node: `False`

The LLMMarkdownComposer node is designed to transform text input into well-structured Markdown documents. It leverages a language model to interpret and format the given text according to specified classifiers and any additional directions, ensuring the output adheres to Markdown standards while incorporating all provided data.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for generating the Markdown document. This model processes the input text, classifiers, and any extra directions to produce the structured output.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `LLM_MODEL`
- **`text_input`**
    - The primary text input that will be transformed into a Markdown document. This input serves as the base content for the document generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`classifier_list`**
    - A comma-separated list of classifiers that guide the language model in structuring the Markdown document. These classifiers help tailor the output to specific formatting or content requirements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`extra_directions`**
    - Additional instructions or guidelines provided to the language model to influence the generation of the Markdown document. These directions can specify further formatting or content considerations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`markdown_output`**
    - Comfy dtype: `STRING`
    - The generated Markdown document, structured and formatted according to the input text, classifiers, and any extra directions provided.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMMarkdownComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "text_input": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Data..."}),
                "classifier_list": ("STRING", {"multiline": False, "dynamicPrompts": False}),
            },
            "optional": {
                "extra_directions": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Extra directions for the LLM to follow..."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown_output",)

    FUNCTION = "compose_markdown"
    CATEGORY = "SALT/Llama-Index/Tools/Markdown"

    def compose_markdown(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = (
            f"{text_input}\n\n###\n\n"
            "Given the above text, create a valid Markdown document utilizing *all* of the data; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the Markdown output is well-structured, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
