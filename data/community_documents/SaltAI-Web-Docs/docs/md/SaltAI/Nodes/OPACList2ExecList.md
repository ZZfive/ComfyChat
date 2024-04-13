# Keyframing Convert Iterative Execution List
## Documentation
- Class name: `OPACList2ExecList`
- Category: `SALT/Animation/Keyframing`
- Output node: `False`

The OPACList2ExecList node is designed to transform a list input into a list output, facilitating iterative execution across elements. This conversion process is essential for enabling sequential processing of list items in animation keyframing contexts.
## Input types
### Required
- **`list_input`**
    - The 'list_input' parameter is the primary input for the node, representing the list to be converted. It plays a crucial role in the node's operation by serving as the source data for the iterative execution process.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The output is a single float value derived from the input list, intended for further processing or analysis in the context of animation keyframing.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPACList2ExecList:
    """
        Converts a list to a list output (iterative execution list)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = "SALT/Animation/Keyframing"

    def convert(self, list_input):
        return (list_input, )

```
