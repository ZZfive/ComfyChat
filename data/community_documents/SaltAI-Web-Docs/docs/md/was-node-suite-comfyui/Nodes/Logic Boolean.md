# Logic Boolean
## Documentation
- Class name: `Logic Boolean`
- Category: `WAS Suite/Logic`
- Output node: `False`

The Logic Boolean node is designed to process a single boolean input and return it unchanged. This node serves as a fundamental building block in logical operations, allowing for the straightforward passing of boolean values through a logical processing pipeline.
## Input types
### Required
- **`boolean_number`**
    - The boolean input to be processed. This parameter is essential for the operation of the node as it determines the boolean value that will be passed through the logical processing pipeline.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - The boolean input converted to a numerical representation. This output is significant as it represents the boolean value processed by the node, converted to a number for further processing.
    - Python dtype: `int`
- **`int`**
    - Comfy dtype: `INT`
    - The boolean input converted to an integer representation. This output underscores the conversion of the boolean value to an integer, facilitating its use in numerical operations.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [Image Input Switch](../../was-node-suite-comfyui/Nodes/Image Input Switch.md)



## Source code
```python
class WAS_Boolean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean_number": ("FLOAT", {"default":1, "min":0, "max":1, "step":1}),
            }
        }

    RETURN_TYPES = ("NUMBER","INT")
    FUNCTION = "return_boolean"

    CATEGORY = "WAS Suite/Logic"

    def return_boolean(self, boolean_number=1):
        return (int(round(boolean_number)), int(round(boolean_number)))

```
