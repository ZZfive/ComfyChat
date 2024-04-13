# LLM Message
## Documentation
- Class name: `LLMChatMessages`
- Category: `SALT/Llama-Index/Messages`
- Output node: `False`

The node is designed to prepare chat messages based on a given prompt and the role of the message (either from the system or the user). It encapsulates the functionality to generate structured chat messages that can be further utilized in chat-based applications or systems.
## Input types
### Required
- **`prompt`**
    - This parameter represents the text of the chat message. It is crucial for generating the content of the chat message, serving as the basis for the message's content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`role`**
    - Specifies the role of the chat message, either as a system-generated message or a user-generated message. This affects how the message is interpreted and displayed in the context of a chat conversation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`llm_message`**
    - Comfy dtype: `LIST`
    - A list of chat messages structured with specified roles and content, ready for use in chat-based interactions.
    - Python dtype: `List[ChatMessage]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatMessages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "prompt"}),
                "role": (["SYSTEM", "USER"],),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def prepare_messages(self, prompt, role):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM if role == "SYSTEM" else MessageRole.USER, content=prompt ),
        ]
        return (messages,)

```
