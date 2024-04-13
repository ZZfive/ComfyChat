# Any To String (mtb)
## Documentation
- Class name: `Any To String (mtb)`
- Category: `mtb/converters`
- Output node: `False`

This node is designed to convert various types of input into their string representations, making it versatile for handling different data formats.
## Input types
### Required
- **`input`**
    - The input parameter accepts any data type, aiming to convert it into a string representation. Its versatility allows for a wide range of data types to be processed, making the node highly adaptable.
    - Comfy dtype: `*`
    - Python dtype: `Union[str, torch.Tensor, PIL.Image.Image, np.ndarray, dict, Any]`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is a string representation of the input, providing a textual description of the input's type and characteristics.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnyToString:
    """Tries to take any input and convert it to a string"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"input": ("*")},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "do_str"
    CATEGORY = "mtb/converters"

    def do_str(self, input):
        if isinstance(input, str):
            return (input,)
        elif isinstance(input, torch.Tensor):
            return (f"Tensor of shape {input.shape} and dtype {input.dtype}",)
        elif isinstance(input, Image.Image):
            return (f"PIL Image of size {input.size} and mode {input.mode}",)
        elif isinstance(input, np.ndarray):
            return (f"Numpy array of shape {input.shape} and dtype {input.dtype}",)

        elif isinstance(input, dict):
            return (f"Dictionary of {len(input)} items, with keys {input.keys()}",)

        else:
            log.debug(f"Falling back to string conversion of {input}")
            return (str(input),)

```
