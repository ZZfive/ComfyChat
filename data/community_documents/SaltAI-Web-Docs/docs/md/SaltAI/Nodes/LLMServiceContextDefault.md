# LLM Service Context
## Documentation
- Class name: `LLMServiceContextDefault`
- Category: `SALT/Llama-Index/Context`
- Output node: `False`

This node is designed to create a default service context for interacting with language model (LLM) services. It simplifies the process of configuring and utilizing LLMs by providing a standardized context setup, which is essential for various LLM operations within the system.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used. This parameter is crucial as it determines the behavior and capabilities of the service context being created.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
## Output types
- **`llm_context`**
    - Comfy dtype: `LLM_CONTEXT`
    - The output is a service context configured with the specified language model. This context is essential for subsequent operations involving the language model.
    - Python dtype: `ServiceContext`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMServiceContextDefault:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
            },
        }

    RETURN_TYPES = ("LLM_CONTEXT",)
    RETURN_NAMES = ("llm_context",)

    FUNCTION = "context"
    CATEGORY = "SALT/Llama-Index/Context"

    def context(self, llm_model):
        service_context = ServiceContext.from_defaults(llm=llm_model)
        return (service_context,)

```
