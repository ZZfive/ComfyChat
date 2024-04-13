# Show Text 🐍
## Documentation
- Class name: `ShowText|pysssss`
- Category: `utils`
- Output node: `True`

The ShowText node is designed to display text within a workflow, optionally updating widget values based on unique identifiers and additional PNG information. It serves as a utility for visual feedback and dynamic text display in user interfaces.
## Input types
### Required
- **`text`**
    - The primary text to be displayed. This is a required input that directly influences the content shown by the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The text input to the node, returned as a string.
    - Python dtype: `str`
- **`ui`**
    - The UI representation of the text to be displayed.
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [Text Concatenate](../../was-node-suite-comfyui/Nodes/Text Concatenate.md)
    - Efficient Loader
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [CR Image Output](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Image Output.md)
    - SavePromptToFile



## Source code
```python
class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def notify(self, text, unique_id = None, extra_pnginfo=None):
        if unique_id and extra_pnginfo and "workflow" in extra_pnginfo[0]:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [text]
        return {"ui": {"text": text}, "result": (text,)}

```
