# LLM HTML Composer
## Documentation
- Class name: `LLMHtmlComposer`
- Category: `SALT/Llama-Index/Tools/HTML`
- Output node: `False`

The LLMHtmlComposer node is designed to generate HTML content based on textual input, utilizing a language model to interpret and structure the input according to specified classifiers and additional directions. It supports creating either a full HTML page or an HTML snippet, depending on the mode selected.
## Input types
### Required
- **`llm_model`**
    - The language model used for generating HTML content from the given text input, utilizing specified classifiers.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `object`
- **`text_input`**
    - The textual content to be transformed into HTML format, serving as the base for the HTML generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`classifier_list`**
    - A list of classifiers that guide the HTML generation process, ensuring the content is structured according to these specified categories.
    - Comfy dtype: `STRING`
    - Python dtype: `List[str]`
### Optional
- **`extra_directions`**
    - Additional instructions to further guide the HTML generation process, allowing for more customized output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`composer_mode`**
    - Determines whether the output will be a full HTML page document or an HTML snippet, affecting the structure of the generated content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`html_output`**
    - Comfy dtype: `STRING`
    - The generated HTML content, structured according to the input text, classifiers, and any additional directions provided.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMHtmlComposer:
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
                "composer_mode": (["full_markup", "blocked_markup"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("html_output",)

    FUNCTION = "compose_html"
    CATEGORY = "SALT/Llama-Index/Tools/HTML"

    def compose_html(self, llm_model, text_input, classifier_list, extra_directions="", composer_mode="full_markup"):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        markup_style = "full HTML page document" if composer_mode == "full_markup" else "HTML snippet (without html, head, body or any extraneous containers)"
        prompt = (
            f"{text_input}\n\n###\n\n"
            f"Given the above text, create a valid {markup_style} utilizing *all* of the data, intact; "
            f"using the following classifiers: {classifier_list}.\n\n"
            f"{extra_directions}\n\n"
            "Please ensure the HTML output is well-structured, valid, and does not omit any data."
        )
        
        response = llm_model.complete(prompt)
        
        return (response.text,)

```
