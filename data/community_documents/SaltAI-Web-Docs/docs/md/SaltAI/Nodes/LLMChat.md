# LLM Chat
## Documentation
- Class name: `LLMChat`
- Category: `SALT/Llama-Index/Querying`
- Output node: `False`

The LLMChat node facilitates interactive communication with a language model, allowing users to send messages and receive responses. This node acts as an interface for querying the language model in a chat-like manner, making it suitable for applications requiring conversational interactions.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used for chatting. It determines the capabilities and limitations of the chat interactions.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `LLM_MODEL`
- **`message`**
    - A list of messages to be sent to the language model. The model processes these messages to generate responses.
    - Comfy dtype: `LIST`
    - Python dtype: `LIST`
## Output types
- **`response`**
    - Comfy dtype: `STRING`
    - The language model's response to the input messages, encapsulated as a string.
    - Python dtype: `STRING`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "message": ("LIST", ),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("response", )

    FUNCTION = "chat"
    CATEGORY = "SALT/Llama-Index/Querying"

    def chat(self, llm_model, message):
        response = llm_model.chat(message)
        pprint(response, indent=4)
        return (response.message.content, )

```
