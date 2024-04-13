# LLM Semantics Splitter Node Parser
## Documentation
- Class name: `LLMSemanticSplitterNodeParser`
- Category: `SALT/Llama-Index/NodeParsing`
- Output node: `False`

This node is designed to parse documents using a semantic splitting approach, leveraging a language model to understand and segment the content semantically. It aims to enhance the structure and accessibility of large text corpora by identifying meaningful segments and metadata, facilitating more efficient indexing and retrieval.
## Input types
### Required
- **`llm_documents`**
    - Represents the documents to be parsed. It is crucial for the semantic analysis and splitting process, serving as the primary data source.
    - Comfy dtype: `LLM_DOCUMENTS`
    - Python dtype: `tuple`
- **`llm_embed_model`**
    - Specifies the language model used for embedding the documents. It plays a key role in understanding the semantic structure of the text.
    - Comfy dtype: `LLM_EMBED_MODEL`
    - Python dtype: `tuple`
### Optional
- **`buffer_size`**
    - Determines the size of the buffer used during parsing. A larger buffer can improve performance but requires more memory.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sentence_splitter`**
    - Optional custom sentence splitter to use. It allows for customization of the sentence splitting process, which can be beneficial for specific languages or document types.
    - Comfy dtype: `LLM_SENTENCE_SPLITTER`
    - Python dtype: `tuple`
- **`include_metadata`**
    - Controls whether metadata is included in the parsing results. Including metadata can enrich the semantic understanding of the documents.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_prev_next_rel`**
    - Determines whether to include relationships between segments, such as previous and next segment links. This can enhance navigation and understanding of the document structure.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`llm_node_parser`**
    - Comfy dtype: `LLM_NODE_PARSER`
    - The result of the semantic parsing process, encapsulating the structured and semantically segmented documents.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSemanticSplitterNodeParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
                "llm_embed_model": ("LLM_EMBED_MODEL",),
            },
            "optional": {
                "buffer_size": ("INT", {"default": 1, "min": 1}),
                "sentence_splitter": ("LLM_SENTENCE_SPLITTER",),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "include_prev_next_rel": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_NODE_PARSER",)
    RETURN_NAMES = ("llm_node_parser",)

    FUNCTION = "semantic_nodes"
    CATEGORY = "SALT/Llama-Index/NodeParsing"

    def semantic_nodes(self, llm_documents, llm_embed_model, buffer_size=1, sentence_splitter=None, include_metadata=True, include_prev_next_rel=True):
        parser = SemanticSplitterNodeParser(
            embed_model=llm_embed_model,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
        return (parser.build_semantic_nodes_from_documents(llm_documents, show_progress=True), )

```
