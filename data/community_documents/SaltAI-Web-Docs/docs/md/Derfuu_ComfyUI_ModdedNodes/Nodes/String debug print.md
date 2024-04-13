# String debug print
## Documentation
- Class name: `String debug print`
- Category: `Derfuu_Nodes/Debug`
- Output node: `True`

This node is designed for debugging purposes, allowing users to print specified text strings prefixed with a custom label to the console. It facilitates the observation and verification of string values during the execution of a node network.
## Input types
### Required
- **`Prefix`**
    - Specifies a custom label to prefix the printed text, aiding in the identification and categorization of the output during debugging.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Text`**
    - The text string to be printed. This input allows users to observe specific string values within the node network.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugNodeString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Prefix": field.STRING,
                "Text": field.STRING,
            }
        }

    RETURN_TYPES = ()
    CATEGORY = TREE_DEBUG
    FUNCTION = "print_values"
    OUTPUT_NODE = True

    def print_values(self, Text, Prefix):
        print(f"{Prefix}: {Text}", sep="\n")
        return (None,)

```
