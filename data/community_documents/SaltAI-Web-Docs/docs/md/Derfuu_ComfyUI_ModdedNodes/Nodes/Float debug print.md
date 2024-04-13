# Float debug print
## Documentation
- Class name: `Float debug print`
- Category: `Derfuu_Nodes/Debug`
- Output node: `True`

This node is designed for debugging purposes, specifically for printing floating-point values alongside a custom prefix to the console. It facilitates the tracking and verification of float values during the execution of a program.
## Input types
### Required
- **`Prefix`**
    - The Prefix parameter allows users to specify a custom prefix that precedes the floating-point value in the console output, aiding in the identification and categorization of printed values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Value`**
    - The Value parameter represents the floating-point number to be printed. It is crucial for debugging, as it allows the inspection of float values at various points in the program's execution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugNodeFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Prefix": field.STRING,
                "Value": field.FLOAT,
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
