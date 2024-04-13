# LLM JSON Composer
## Documentation
- Class name: `LLMJsonComposer`
- Category: `SALT/Llama-Index/Tools/JSON`
- Output node: `False`

The LLMJsonComposer node is designed to generate well-formed JSON objects from textual inputs using specified classifiers and additional directions. It leverages a language model to interpret and structure the input text into JSON format, ensuring the output is correctly formatted and comprehensive.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to use for generating the JSON object. It plays a crucial role in interpreting the input text and structuring it into a valid JSON format.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `LLM_MODEL`
- **`text_input`**
    - The textual data to be transformed into a JSON object. This input is the primary content that will be analyzed and structured by the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`classifier_list`**
    - A list of classifiers used to guide the JSON composition process, ensuring the output adheres to specified criteria.
    - Comfy dtype: `STRING`
    - Python dtype: `List[str]`
### Optional
- **`extra_directions`**
    - Additional instructions for the language model to follow when composing the JSON object, allowing for more tailored outputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The well-formed JSON object generated from the input text, structured according to the specified classifiers and additional directions.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMJsonComposer:
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
    RETURN_NAMES = ("json_output",)

    FUNCTION = "compose_json"
    CATEGORY = "SALT/Llama-Index/Tools/JSON"

    def compose_json(self, llm_model, text_input, classifier_list, extra_directions=""):
        classifier_list = [item.strip() for item in classifier_list.split(",") if item.strip()]
        prompt = f"{text_input}\n\n###\n\nGiven the above text, create a valid JSON object utilizing *all* of the data; using the following classifiers: {classifier_list}.\n\n{extra_directions}\n\nPlease ensure the JSON output is properly formatted, and does not omit any data."
        response = llm_model.complete(prompt)
        return (response.text,)

```
