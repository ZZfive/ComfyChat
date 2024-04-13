# Text Input Switch
## Documentation
- Class name: `Text Input Switch`
- Category: `WAS Suite/Logic`
- Output node: `False`

The Text Input Switch node is designed to selectively output one of two text inputs based on a boolean condition. It serves as a logical switch that can dynamically alter the flow of text data within a pipeline, based on specified conditions.
## Input types
### Required
- **`text_a`**
    - The first text input option. This input, along with 'text_b', forms the pair between which the node chooses based on the boolean condition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_b`**
    - The second text input option. This input, along with 'text_a', forms the pair between which the node chooses based on the boolean condition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`boolean_number`**
    - A numeric input that determines which text input ('text_a' or 'text_b') to output. Typically, a value of 1 selects 'text_a', while any other value selects 'text_b'.
    - Comfy dtype: `NUMBER`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the selected text input based on the boolean condition. This allows for dynamic control over which text data is passed forward in the pipeline.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Text_Input_Switch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_a": (TEXT_TYPE, {"forceInput": (True if TEXT_TYPE == 'STRING' else False)}),
                "text_b": (TEXT_TYPE, {"forceInput": (True if TEXT_TYPE == 'STRING' else False)}),
                "boolean_number": ("NUMBER",),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "text_input_switch"

    CATEGORY = "WAS Suite/Logic"

    def text_input_switch(self, text_a, text_b, boolean_number=1):

        if int(round(boolean_number)) == 1:
            return (text_a, )
        else:
            return (text_b, )

```
