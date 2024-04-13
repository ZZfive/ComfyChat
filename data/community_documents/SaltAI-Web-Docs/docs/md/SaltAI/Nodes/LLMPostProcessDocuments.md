# LLM Post Process Documents
## Documentation
- Class name: `LLMPostProcessDocuments`
- Category: `SALT/Llama-Index/Processing`
- Output node: `False`

The LLMPostProcessDocuments node is designed for refining a collection of documents based on specified inclusion and exclusion criteria. It filters documents by ensuring they contain any of the required keywords while not containing any of the excluded keywords, thereby tailoring the document set to more closely match user specifications.
## Input types
### Required
- **`llm_documents`**
    - The primary dataset of documents to be processed. This parameter is essential for the operation as it provides the base content that will be filtered according to the specified keywords.
    - Comfy dtype: `LLM_DOCUMENTS`
    - Python dtype: `List[Document]`
### Optional
- **`required_keywords`**
    - A comma-separated list of keywords that must be present in a document for it to be included in the output. This parameter allows for the targeting of documents that meet specific content criteria.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_keywords`**
    - A comma-separated list of keywords that, if present in a document, will result in its exclusion from the output. This parameter is used to remove documents containing undesired content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The filtered set of documents that meet the inclusion and exclusion criteria specified by the input parameters.
    - Python dtype: `List[Document]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMPostProcessDocuments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
            },
            "optional": {
                "required_keywords": ("STRING", {}),
                "exclude_keywords": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "process_documents"
    CATEGORY = "SALT/Llama-Index/Processing"

    def process_documents(self, llm_documents, required_keywords=[], exclude_keywords=[]):

        if required_keywords.strip():
            required = [ext.strip() for ext in required_keywords.split(",") if ext.strip()]
        else:
            required = None

        if exclude_keywords.strip():
            excluded = [pattern.strip() for pattern in exclude_keywords.split(",") if pattern.strip()]
        else:
            excluded = None

        if required or excluded:
            llm_documents = [doc for doc in llm_documents if not set(required).isdisjoint(doc.keywords) and set(excluded).isdisjoint(doc.keywords)]

        return (llm_documents,)

```
