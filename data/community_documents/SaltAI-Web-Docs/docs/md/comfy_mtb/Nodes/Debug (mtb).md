# Debug (mtb)
## Documentation
- Class name: `Debug (mtb)`
- Category: `mtb/debug`
- Output node: `True`

The Debug node is designed for experimental purposes to facilitate debugging of various Comfy values. It aims to provide a flexible debugging tool with planned support for additional types and widgets, enhancing the development and troubleshooting process within the Comfy environment.
## Input types
### Required
- **`output_to_console`**
    - Determines whether the debug output should be printed to the console. This affects the node's execution by enabling or disabling console logging of the debug information.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`ui`**
    - Contains the debugging information processed from the input, including base64-encoded images and text data, structured for UI display.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Debug:
    """Experimental node to debug any Comfy values, support for more types and widgets is planned"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"output_to_console": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ()
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(self, output_to_console, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": []},
            # "result": ("A"),
        }

        processors = {
            torch.Tensor: process_tensor,
            list: process_list,
            dict: process_dict,
            bool: process_bool,
        }
        if output_to_console:
            print("bouh!")

        for anything in kwargs.values():
            processor = processors.get(type(anything), process_text)
            processed_data = processor(anything)

            for ui_key, ui_value in processed_data.items():
                output["ui"][ui_key].extend(ui_value)
            # log.debug(
            #     f"Processed input {k}, found {len(processed_data.get('b64_images', []))} images and {len(processed_data.get('text', []))} text items."
            # )

        return output

```
