# LLM Summary Index
## Documentation
- Class name: `LLMSummaryIndex`
- Category: `SALT/Llama-Index/Indexing`
- Output node: `False`

The LLMSummaryIndex node is designed for indexing large language model (LLM) documents by summarizing their content, facilitating efficient retrieval and search operations within the indexed data.
## Input types
### Required
- **`llm_documents`**
    - Represents the documents to be indexed. It is crucial for creating a searchable summary index of the provided documents.
    - Comfy dtype: `LLM_DOCUMENTS`
    - Python dtype: `Tuple[str]`
- **`llm_context`**
    - Provides context or additional settings for the indexing process, influencing how documents are summarized and indexed.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `str`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The output is a searchable index created from the summarized documents, enabling efficient retrieval based on content relevance.
    - Python dtype: `VectorStoreIndex`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSummaryIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
                "llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_documents, llm_context):
        index = SummaryIndex.from_documents(llm_documents, service_context=llm_context)
        return (index,)

```
