# Salt Flow Input
## Documentation
- Class name: `SaltInput`
- Category: `SALT/IO`
- Output node: `True`

The SaltInput node is designed to handle various types of user inputs, including strings, numbers, booleans, and files. It validates the input against allowed values if specified, converts it to the appropriate data type, and prepares it for further processing within a workflow. This node is essential for capturing and preprocessing user inputs in a flexible and dynamic manner.
## Input types
### Required
- **`input_name`**
    - Specifies the name of the input, serving as an identifier within the workflow.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`input_desc`**
    - Provides a description for the input, offering context or instructions for the user.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`input_type`**
    - Defines the data type of the input, such as STRING, FLOAT, INT, BOOLEAN, IMAGE, or MASK. This selection is crucial as it dictates how the input should be processed, validated, and converted, directly influencing the node's execution and the accuracy of the results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`input_value`**
    - The actual value provided by the user, which will be converted and validated according to the input_type.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`input_image`**
    - An optional image file provided by the user, relevant when the input_type is IMAGE.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[str]`
- **`input_mask`**
    - An optional mask file provided by the user, relevant when the input_type is MASK.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[str]`
- **`input_allowed_values`**
    - A comma-separated list of allowed values for the input, used to validate the user's input against a predefined set of acceptable values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`user_override_required`**
    - A flag indicating whether the user's input must override a default value, enhancing the flexibility and control over the input's behavior.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`value`**
    - Comfy dtype: `*`
    - The processed and validated input value, ready for use in subsequent nodes or operations.
    - Python dtype: `str`
- **`ui`**
    - A dictionary containing UI-related information for the input, including its name, description, and whether it's an asset (e.g., an image or mask).
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {}),
                "input_desc": ("STRING", {}),
                "input_type": (["STRING", "FLOAT", "INT", "BOOLEAN", "IMAGE", "MASK", "SEED"],),
                "input_value": ("STRING", {"multiline": True, "dynamicPrompts": False})
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_mask": ("MASK",),
                "input_allowed_values": ("STRING", {"default": ""}),
                "user_override_required": ("BOOLEAN", {"default": False})
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("value",)

    FUNCTION = "input"
    CATEGORY = "SALT/IO"

    def input(
        self,
        input_name,
        input_desc,
        input_value,
        input_type,
        input_image=None,
        input_mask=None,
        input_allowed_values="",
        user_override_required=False,
        unique_id=0,
    ):
        src_image = None
        is_asset = False

        if input_type in ["IMAGE", "MASK"]:
            is_asset = True

        ui = {
            "ui": {
                "salt_input": [{
                    "id": unique_id,
                    "name": input_name or "input_" + str(unique_id),
                    "description": input_desc or "",
                    "asset": is_asset or False,
                    "type": input_type or "string",
                    "value": input_value or "",
                }]
            }
        }

        if is_asset:
            if isinstance(input_image, torch.Tensor):
                # Input `IMAGE` is provided, so we act like a passthrough
                return (input_image, ui)
            elif isinstance(input_mask, torch.Tensor):
                # Input `MASK` is provided, so we act like a passthrough
                return (input_mask, ui)
            elif input_value.strip():
                # Load image from path from input_value
                try:
                    src_image = Image.open(input_value.strip()).convert("RGBA")
                except Exception as e:
                    print(f"Error loading image from specified path {input_value}: {e}")

            if src_image:
                if input_type == "MASK":
                    # If it's a mask and the image has an alpha channel, extract it
                    if src_image.mode == "RGBA":
                        alpha_channel = src_image.split()[-1]
                        src_image = pil2mask(alpha_channel.convert("L"))
                    # If no alpha channel, convert the whole image to grayscale as a mask (could be bitwise representation)
                    else:
                        src_image = pil2mask(src_image.convert("L"))
                elif input_type == "IMAGE":
                    # Ensure image is in RGB for `IMAGE`` data type
                    src_image = pil2tensor(src_image.convert("RGB"))

            else:
                # Gracefully allow execution to continue, provided a black image (to hopefully signal issue?)
                print("[WARNING] Unable to determine IMAGE or MASK to load!")
                print("[WARNING] Returning image blank")
                src_blank = Image.new("RGB", (512, 512), (0, 0, 0))
                if input_type == "IMAGE":
                    src_image = pil2tensor(src_blank)
                else:
                    src_image = pil2mask(src_blank)

            return (src_image, ui)

        # We're still here? We must be dealing with a primitive value
        if input_allowed_values != "" and input_value.strip() not in [o.strip() for o in input_allowed_values.split(',')]:
            raise ValueError('The provided input is not a supported value')


        out = ""
        match input_type:
            case "STRING":
                out = str(input_value)
            case "INT":
                out = int(input_value)
            case "SEED":
                out = int(input_value)
            case "FLOAT":
                out = float(input_value)
            case "BOOLEAN":
                out = bool_str(input_value)
            case _:
                out = input_value

        # Log value to console
        print(f"[SaltInput_{unique_id}] `{input_name}` ({input_type}) Value:")
        print(out)

        return (out, ui)

```
