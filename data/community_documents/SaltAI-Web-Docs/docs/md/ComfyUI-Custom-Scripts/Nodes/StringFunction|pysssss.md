# String Function 🐍
## Documentation
- Class name: `StringFunction|pysssss`
- Category: `utils`
- Output node: `True`

The String Function node provides utilities for manipulating strings in various ways, such as appending, replacing, and optionally adding an additional string. It supports operations on multi-line strings and can be configured to tidy up HTML tags.
## Input types
### Required
- **`action`**
    - Specifies the string manipulation action to perform, either appending or replacing content in the input strings.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`tidy_tags`**
    - Determines whether to clean up HTML tags from the input strings, with options to either do so or not.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`text_a`**
    - The primary text input for manipulation. Supports multi-line strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_b`**
    - The secondary text input for manipulation. Supports multi-line strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`text_c`**
    - An optional third text input for manipulation. Supports multi-line strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The result of the string manipulation operation.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["append", "replace"], {}),
                "tidy_tags": (["yes", "no"], {}),
                "text_a": ("STRING", {"multiline": True}),
                "text_b": ("STRING", {"multiline": True}),
            },
            "optional": {
                "text_c": ("STRING", {"multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "exec"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def exec(self, action, tidy_tags, text_a, text_b, text_c=""):
        # Converted inputs are sent as the string of 'undefined' if not connected
        if text_a == "undefined":
            text_a = ""
        if text_b == "undefined":
            text_b = ""
        if text_c == "undefined":
            text_c = ""
            
        tidy_tags = tidy_tags == "yes"
        out = ""
        if action == "append":
            out = (", " if tidy_tags else "").join(filter(None, [text_a, text_b, text_c]))
        else:
           if text_c is None:
               text_c = ""
           if text_b.startswith("/") and text_b.endswith("/"):
               regex = text_b[1:-1]
               out = re.sub(regex, text_c, text_a)
           else:
               out = text_a.replace(text_b, text_c)
        if tidy_tags:
            out = out.replace("  ", " ").replace(" ,", ",").replace(",,", ",").replace(",,", ",")
        return {"ui": {"text": (out,)}, "result": (out,)}

```
