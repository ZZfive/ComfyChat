# LLM Tree Index
## Documentation
- Class name: `LLMTreeIndex`
- Category: `SALT/Llama-Index/Indexing`
- Output node: `False`

The LLMTreeIndex node is designed for creating an index structure from a set of nodes and a service context, optionally configuring the number of children per node and whether to build the tree structure. This index facilitates efficient organization and retrieval of information.
## Input types
### Required
- **`llm_nodes`**
    - Represents the nodes to be indexed, serving as the primary data structure for the indexing process.
    - Comfy dtype: `LLM_NODES`
    - Python dtype: `tuple`
- **`service_context`**
    - Provides the context required for the service, ensuring that the indexing process is aligned with specific service needs.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `tuple`
### Optional
- **`num_children`**
    - Specifies the maximum number of children each node in the tree can have, influencing the breadth of the tree structure.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`build_tree`**
    - Determines whether to construct the tree structure during the indexing process, enabling hierarchical organization of nodes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The resulting index structure, optimized for efficient data retrieval and organization.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMTreeIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_nodes": ("LLM_NODES",),
                "service_context": ("LLM_CONTEXT",),
            },
            "optional": {
                "num_children": ("INT", {"default": 10}),
                "build_tree": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_nodes, service_context, num_children=10, build_tree=True):
        index = TreeIndex(
            nodes=llm_nodes,
            num_children=num_children,
            build_tree=build_tree,
            use_async=False,
            show_progress=True,
            service_context=service_context,
        )
        return (index,)

```
