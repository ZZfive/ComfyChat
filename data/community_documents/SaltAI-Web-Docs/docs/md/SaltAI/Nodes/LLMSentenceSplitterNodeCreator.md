# LLM Setence Splitter Node Creator
## Documentation
- Class name: `LLMSentenceSplitterNodeCreator`
- Category: `SALT/Llama-Index/Nodes`
- Output node: `False`

This node is designed to split large language model (LLM) documents into smaller, manageable nodes based on sentence boundaries. It facilitates the processing and indexing of large text corpora by breaking them down into more digestible pieces.
## Input types
### Required
- **`llm_documents`**
    - Represents the large language model documents to be split. It is essential for determining the text corpus that will undergo the sentence splitting process.
    - Comfy dtype: `LLM_DOCUMENTS`
    - Python dtype: `Tuple[str]`
### Optional
- **`chunk_size`**
    - Defines the maximum size of each chunk or node created from the original documents. It plays a crucial role in controlling the granularity of the splitting process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`chunk_overlap`**
    - Specifies the number of characters that overlap between consecutive chunks. This parameter ensures continuity and context preservation across the split nodes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`llm_nodes`**
    - Comfy dtype: `LLM_NODES`
    - The output consists of smaller, sentence-based nodes derived from the original LLM documents. These nodes are ready for further processing or indexing.
    - Python dtype: `Tuple[Node]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSentenceSplitterNodeCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 1024, "min": 1}),
                "chunk_overlap": ("INT", {"default": 20, "min": 0}),
            },
        }

    RETURN_TYPES = ("LLM_NODES",)
    RETURN_NAMES = ("llm_nodes",)

    FUNCTION = "create_nodes"
    CATEGORY = "SALT/Llama-Index/Nodes"

    def create_nodes(self, llm_documents, chunk_size=1024, chunk_overlap=20):
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(llm_documents, show_progress=False)        
        return (nodes,)

```
