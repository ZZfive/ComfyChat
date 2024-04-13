# String
## Documentation
- Class name: `String`
- Category: `Bmad`
- Output node: `False`

The StringNode is designed to simply pass through the input string it receives, making it a straightforward utility for string manipulation or forwarding within a pipeline.
## Input types
### Required
- **`inStr`**
    - Represents the input string to be passed through. Its primary role is to serve as the direct input that the node will output unchanged, emphasizing its utility in scenarios where the integrity of the input string must be preserved.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the same string that was input, underscoring the node's function as a pass-through mechanism.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pass_it"
    CATEGORY = "Bmad"

    def pass_it(self, inStr):
        return (inStr,)

```
