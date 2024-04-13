# SomethingToString
## Documentation
- Class name: `SomethingToString`
- Category: `KJNodes`
- Output node: `False`

The SomethingToString node is designed to convert various basic data types (integers, floats, and booleans) into their string representations. This node emphasizes simplicity and utility, providing a straightforward way to transform numerical or boolean inputs into a format that can be easily displayed or processed as text.
## Input types
### Required
- **`input`**
    - The 'input' parameter accepts integers, floats, and booleans, converting them into their string representations. This conversion facilitates the easy display or further processing of these values as text.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool]`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns the string representation of the input value.
    - Python dtype: `Tuple[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SomethingToString:
    @classmethod
    
    def INPUT_TYPES(s):
     return {
        "required": {
        "input": ("*", {"forceinput": True, "default": ""}),
    },
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stringify"
    CATEGORY = "KJNodes"

    def stringify(self, input):
        if isinstance(input, (int, float, bool)):   
            stringified = str(input)
            print(stringified)
        else:
            return
        return (stringified,)

```
