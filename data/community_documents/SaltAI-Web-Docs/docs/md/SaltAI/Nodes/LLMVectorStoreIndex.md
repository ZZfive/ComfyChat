# LLM Vector Store Index
## Documentation
- Class name: `LLMVectorStoreIndex`
- Category: `SALT/Llama-Index/Indexing`
- Output node: `False`

The LLMVectorStoreIndex node is designed for indexing large language model (LLM) documents within a vector space, facilitating efficient retrieval and similarity searches. It abstracts the complexity of transforming textual documents into a vectorized format that can be easily queried.
## Input types
### Required
- **`llm_documents`**
    - Represents the documents to be indexed. It is crucial for creating a searchable vector space of the documents.
    - Comfy dtype: `LLM_DOCUMENTS`
    - Python dtype: `Tuple[str]`
- **`llm_context`**
    - Provides the context or settings under which the indexing service operates, affecting how documents are processed and indexed.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `str`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The output is an index structure that enables efficient querying of the large language model documents based on similarity.
    - Python dtype: `VectorStoreIndex`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMVectorStoreIndex:
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
        index = VectorStoreIndex.from_documents(llm_documents, service_context=llm_context)
        return (index,)

```
