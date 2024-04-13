# Int debug print
## Documentation
- Class name: `Int debug print`
- Category: `Derfuu_Nodes/Debug`
- Output node: `True`

The node is designed for debugging purposes, specifically for printing integer values alongside a custom prefix to the console. This facilitates tracking and verifying the flow of integer data through the node network.
## Input types
### Required
- **`Prefix`**
    - Specifies a custom prefix to be printed before the integer value, aiding in distinguishing between different debug prints.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Value`**
    - The integer value to be printed. This is the primary data the node is designed to debug.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugNodeInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Prefix": field.STRING,
                "Value": field.INT,
            }
        }

    RETURN_TYPES = ()
    CATEGORY = TREE_DEBUG
    FUNCTION = "print_values"
    OUTPUT_NODE = True

    def print_values(self, Value, Prefix):
        print(f"{Prefix}: {Value}", sep="\n")
        return (None,)

```
