# Text
## Documentation
- Class name: `Text`
- Category: `Derfuu_Nodes/Variables`
- Output node: `False`

The Text node is designed to handle and process string inputs, allowing for the manipulation and transformation of text data within a computational graph. It serves as a fundamental building block for text-based operations, enabling the creation, modification, and analysis of strings.
## Input types
### Required
- **`Text`**
    - This parameter represents the text input to the node. It is essential for text-based operations, serving as the primary data that will be processed or manipulated by the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - This output represents the processed or manipulated text data. It is the result of the node's operation on the input text, potentially transformed or analyzed in some way.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": field.STRING,
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = TREE_VARIABLE

    def get_value(self, Text):
        return (Text,)

```
