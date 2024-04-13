# LLM Message (Advanced)
## Documentation
- Class name: `LLMChatMessagesAdv`
- Category: `SALT/Llama-Index/Messages`
- Output node: `False`

The LLMChatMessagesAdv node is designed for preparing chat messages by combining system prompts and user inputs into a structured format. This node facilitates the creation of chat message sequences that can be further processed or analyzed within a chatbot or conversational AI system.
## Input types
### Required
- **`system_prompt`**
    - The system prompt represents the initial message or context provided by the system. It sets the stage for the conversation and guides the user's response.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`user_prompt`**
    - The user prompt is the input provided by the user in response to the system prompt. It represents the user's side of the conversation and is crucial for generating a dynamic and interactive chat experience.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_message`**
    - Comfy dtype: `LIST`
    - This output is a list of chat messages structured with roles (system or user) and content, ready for further processing in conversational models or systems.
    - Python dtype: `List[ChatMessage]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatMessagesAdv:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "You are a dog, you cannot speak, only woof, and react as a dog would."}),
                "user_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "plaeholder": "What is your name?"}),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def prepare_messages(self, system_prompt, user_prompt):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt ),
            ChatMessage(role=MessageRole.USER, content=user_prompt ),
        ]
        return (messages,)

```
