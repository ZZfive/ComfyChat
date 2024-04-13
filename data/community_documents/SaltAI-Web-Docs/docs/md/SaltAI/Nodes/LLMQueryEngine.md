# LLM Query Engine
## Documentation
- Class name: `LLMQueryEngine`
- Category: `SALT/Llama-Index/Querying`
- Output node: `False`

The LLMQueryEngine node is designed to process queries using a language model index, allowing for the retrieval of information or answers based on the input query and optional message list. It abstracts the complexity of querying a language model, providing a straightforward interface for obtaining relevant responses.
## Input types
### Required
- **`llm_index`**
    - Specifies the language model index to be used for querying. It is essential for determining the source of information the query will be executed against.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `LLM_INDEX`
### Optional
- **`query`**
    - The user's input query. It's optional but when provided, it enhances the specificity and relevance of the search results.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`llm_message`**
    - An optional list of messages that can be used to provide additional context or information for the query, enhancing the relevance of the results.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
## Output types
- **`results`**
    - Comfy dtype: `STRING`
    - The output of the query engine, typically containing the information or answers retrieved from the language model index based on the input query.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = "SALT/Llama-Index/Querying"

    def query_engine(self, llm_index, query=None, llm_message=None):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))

        if query:
            if query.strip():
                query_components.append("user: " + query)

        pprint(query_components, indent=4)

        query_join = "\n".join(query_components)

        query_engine = llm_index.as_query_engine()
        response = query_engine.query(query_join)
        pprint(response, indent=4)
        return (response.response,)

```
