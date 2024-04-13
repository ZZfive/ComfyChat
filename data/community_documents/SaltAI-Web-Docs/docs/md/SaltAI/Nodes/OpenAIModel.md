# Load OpenAI LLM Model
## Documentation
- Class name: `OpenAIModel`
- Category: `SALT/Llama-Index/Loaders`
- Output node: `False`

The OpenAIModel node is designed to load and initialize models from the OpenAI API, supporting a variety of GPT-3.5 and GPT-4 models. It abstracts the complexity of API interactions and model initialization, providing a straightforward way to access and utilize OpenAI's powerful language models.
## Input types
### Required
- **`model`**
    - Specifies the particular OpenAI model to be loaded, such as various versions of GPT-3.5 and GPT-4, allowing for flexibility in choosing the model that best suits the task at hand.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`api_key`**
    - The API key required for authenticating with the OpenAI API, enabling secure access to the model loading functionality.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_model`**
    - Comfy dtype: `LLM_MODEL`
    - The loaded OpenAI language model, ready for use in generating text or performing other language-based tasks.
    - Python dtype: `OpenAI`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OpenAIModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-instruct",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-16k-0613",
                    "gpt-4-0125-preview",
                    "gpt-4-turbo-preview",
                    "gpt-4-1106-preview",
                    "gpt-4-vision-preview",
                    "gpt-4-1106-vision-preview",
                    "gpt-4",
                    "gpt-4-0613",
                    "gpt-4-32k",
                    "gpt-4-32k-0613"
                ],),
                "api_key": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("llm_model", )

    FUNCTION = "load_model"
    CATEGORY = "SALT/Llama-Index/Loaders"

    def load_model(self, model, api_key):
        openai.api_key = api_key
        llm_model = OpenAI(model=model, api_key=api_key)
        return (llm_model, )

```
