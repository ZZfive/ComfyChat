# Tuple debug print
## Documentation
- Class name: `Tuple debug print`
- Category: `Derfuu_Nodes/Debug`
- Output node: `True`

This node is designed for debugging purposes, specifically to print tuples along with a prefix to the console. It facilitates the observation and verification of tuple values during the development and debugging phases of a project.
## Input types
### Required
- **`Prefix`**
    - The prefix is a string that precedes the tuple values in the print statement, helping to identify or categorize the output in the console.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Values`**
    - The tuple values to be printed. This parameter allows for the inspection of tuple contents, aiding in debugging and development tasks.
    - Comfy dtype: `TUPLE`
    - Python dtype: `tuple`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugNodeTuple:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Prefix": field.STRING,
                "Values": ("TUPLE",),
            }
        }

    RETURN_TYPES = ()
    CATEGORY = TREE_DEBUG
    FUNCTION = "print_values"
    OUTPUT_NODE = True

    def print_values(self, Values, Prefix):
        print(f"{Prefix}: {Values}", sep="\n")
        return (None,)

```
