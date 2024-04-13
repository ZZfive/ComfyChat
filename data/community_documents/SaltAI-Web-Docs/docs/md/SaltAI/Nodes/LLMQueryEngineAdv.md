# LLM Query Engine (Advanced)
## Documentation
- Class name: `LLMQueryEngineAdv`
- Category: `SALT/Llama-Index/Querying`
- Output node: `False`

The LLMQueryEngineAdv node is designed for advanced querying within the Llama-Index ecosystem, allowing for the integration of additional parameters such as top_k and similarity_cutoff to refine search results. It leverages a vector index retriever and a retriever query engine to process and return the most relevant responses based on the input query and message components.
## Input types
### Required
- **`llm_index`**
    - The llm_index parameter is crucial for specifying the index to be used for querying. It enables the node to access and search through the indexed data effectively.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `LLM_INDEX`
### Optional
- **`query`**
    - The query parameter allows users to input their search query. It is processed along with llm_message to form the final query string sent to the query engine.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`llm_message`**
    - The llm_message parameter accepts a list of messages that can be included in the query. These messages are combined with the user's query to enhance the context of the search.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
- **`top_k`**
    - The top_k parameter determines the number of top similar results to retrieve, enhancing the precision of the search results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`similarity_cutoff`**
    - The similarity_cutoff parameter sets a threshold for the similarity score, filtering out results that fall below this value to ensure relevance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`results`**
    - Comfy dtype: `STRING`
    - The results parameter contains the response from the query engine, which includes the most relevant search results based on the input query and additional parameters.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMQueryEngineAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
                "top_k": ("INT", {"default": 10}),
                "similarity_cutoff": ("FLOAT", {"default": 0.7}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = "SALT/Llama-Index/Querying"

    def query_engine(self, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))

        if query and query.strip():
            query_components.append("user: " + query)

        pprint(query_components, indent=4)
        query_join = "\n".join(query_components)

        retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

        response = query_engine.query(query_join)
        pprint(response, indent=4)
        return (response.response,)

```
