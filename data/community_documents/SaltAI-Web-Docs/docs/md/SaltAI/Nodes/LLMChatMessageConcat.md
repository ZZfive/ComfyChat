# LLM Messages Concat
## Documentation
- Class name: `LLMChatMessageConcat`
- Category: `SALT/Llama-Index/Messages`
- Output node: `False`

The LLMChatMessageConcat node is designed to concatenate two lists of chat messages into a single list, facilitating the aggregation of chat message sequences for further processing or analysis.
## Input types
### Required
- **`message_a`**
    - Represents the first list of chat messages to be concatenated. It plays a crucial role in the concatenation process by serving as the initial sequence of messages.
    - Comfy dtype: `LIST`
    - Python dtype: `List[ChatMessage]`
- **`message_b`**
    - Represents the second list of chat messages to be concatenated. It is essential for extending the initial sequence of messages provided by 'message_a' with additional content.
    - Comfy dtype: `LIST`
    - Python dtype: `List[ChatMessage]`
## Output types
- **`llm_message`**
    - Comfy dtype: `LIST`
    - The concatenated list of chat messages, combining the sequences from 'message_a' and 'message_b' into a single, unified list.
    - Python dtype: `List[ChatMessage]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatMessageConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message_a": ("LIST", ),
                "message_b": ("LIST", ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "concat_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def concat_messages(self, message_a, message_b):
        return (message_a + message_b, )

```
