# Text box
## Documentation
- Class name: `Text box`
- Category: `Derfuu_Nodes/Variables`
- Output node: `False`

The Text box node is designed to capture and return multiline string inputs. It serves as a utility for gathering extensive text data, such as paragraphs or multiple lines of text, from the user.
## Input types
### Required
- **`Text`**
    - Represents the multiline string input from the user. It is essential for capturing text data that spans multiple lines, allowing for more detailed and extensive input.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the exact multiline string input received from the user.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - SDXLPromptStyler
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [BatchPromptSchedule](../../ComfyUI_FizzNodes/Nodes/BatchPromptSchedule.md)
    - [Text Concatenate](../../was-node-suite-comfyui/Nodes/Text Concatenate.md)
    - [Text Random Line](../../was-node-suite-comfyui/Nodes/Text Random Line.md)



## Source code
```python
class MultilineStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"
    CATEGORY = TREE_VARIABLE

    def get_value(self, Text):
        return (Text,)

```
