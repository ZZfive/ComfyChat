# String Replace (mtb)
## Documentation
- Class name: `String Replace (mtb)`
- Category: `mtb/string`
- Output node: `False`

The StringReplace node is designed for performing basic string manipulation tasks, specifically replacing occurrences of a substring within a given string with another substring. This functionality is essential in text processing and data cleaning operations, allowing for the modification of text data in a straightforward manner.
## Input types
### Required
- **`string`**
    - The 'string' parameter is the original text in which replacements will be made. It serves as the primary input for the operation, determining the context and content of the manipulation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`old`**
    - The 'old' parameter specifies the substring that should be identified and replaced within the original string. Its presence is crucial for targeting the specific text segments for replacement.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`new`**
    - The 'new' parameter defines the substring that will replace occurrences of the 'old' substring within the original string. This parameter directly influences the outcome of the text manipulation, enabling the customization of the text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is the modified string after all occurrences of the 'old' substring have been replaced with the 'new' substring.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringReplace:
    """Basic string replacement"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "old": ("STRING", {"default": ""}),
                "new": ("STRING", {"default": ""}),
            }
        }

    FUNCTION = "replace_str"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "mtb/string"

    def replace_str(self, string: str, old: str, new: str):
        log.debug(f"Current string: {string}")
        log.debug(f"Find string: {old}")
        log.debug(f"Replace string: {new}")

        string = string.replace(old, new)

        log.debug(f"New string: {string}")

        return (string,)

```
