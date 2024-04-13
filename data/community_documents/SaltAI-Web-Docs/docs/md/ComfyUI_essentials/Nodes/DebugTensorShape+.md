# 🔧 Tensor Shape Debug
## Documentation
- Class name: `DebugTensorShape+`
- Category: `essentials`
- Output node: `True`

This node is designed to analyze and print the shapes of tensors within a given input structure, which can include nested lists, dictionaries, or tensors themselves. It recursively traverses the input to identify and list the shapes of all tensors, providing a clear overview of their dimensions for debugging purposes.
## Input types
### Required
- **`tensor`**
    - The input tensor or a nested structure containing tensors. This parameter is crucial for the node to traverse and identify the shapes of all tensors within the given structure, aiding in debugging tensor dimensions.
    - Comfy dtype: `*`
    - Python dtype: `Union[torch.Tensor, Dict, List]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugTensorShape:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (any, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "essentials"
    OUTPUT_NODE = True

    def execute(self, tensor):
        shapes = []
        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, 'shape'):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)
        
        print(f"\033[96mShapes found: {shapes}\033[0m")

        return (None,)

```
