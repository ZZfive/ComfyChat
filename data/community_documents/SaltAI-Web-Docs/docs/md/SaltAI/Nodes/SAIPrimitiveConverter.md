# Primitive Value Converter
## Documentation
- Class name: `SAIPrimitiveConverter`
- Category: `SALT/Utility/Conversion`
- Output node: `False`

The SAIPrimitiveConverter node is designed to facilitate the conversion of input values between different primitive data types. It supports a wide range of conversions, including string, integer, float, boolean, list, and dictionary types, making it a versatile tool for data type manipulation within workflows.
## Input types
### Required
- **`input_value`**
    - The value to be converted. This parameter is central to the node's functionality, as it determines the input data that will undergo type conversion.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`output_type`**
    - Specifies the desired output data type for the conversion, such as string, integer, float, etc. This parameter directly influences the result of the conversion process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`sub_data_type`**
    - An optional parameter that allows for further specification of the data type for the conversion, enhancing the node's flexibility in handling different data formats.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`index_or_key`**
    - Optionally specifies an index or key for targeted conversion within complex data structures like lists or dictionaries, providing more granular control over the conversion process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`output`**
    - Comfy dtype: `*`
    - The result of the conversion process, which can be of various primitive data types depending on the specified output type.
    - Python dtype: `Any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SAIPrimitiveConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": (WILDCARD,),
                "output_type": (["STRING", "INT", "FLOAT", "BOOLEAN", "LIST", "DICT"],),
            },
            "optional": {
                "sub_data_type": (["ORIGIN", "STRING", "INT", "FLOAT", "BOOLEAN"],),
                "index_or_key": ("STRING", {}),
            },
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("output",)

    FUNCTION = "convert_type"
    CATEGORY = "SALT/Utility/Conversion"

    def convert_type(self, input_value, output_type, sub_data_type="STRING", index_or_key=""):

        def cast_value(value, data_type):
            try:
                if data_type == "ORIGIN":
                    return value
                if data_type == "STRING":
                    return str(value)
                elif data_type == "INT":
                    return int(float(value)) if '.' in value else int(value)
                elif data_type == "FLOAT":
                    return float(value)
                elif data_type == "BOOLEAN":
                    return bool_str(value)
                else:
                    return value
            except Exception as e:
                print(f"[WARNING] {e}")
                return value

        default_values = {
            "STRING": "",
            "INT": 0,
            "FLOAT": 0.0,
            "BOOLEAN": False,
            "LIST": [],
            "DICT": {},
        }

        def process_input_value(input_val):
            if isinstance(input_val, str):
                if ',' in input_val and not any(':' in part for part in input_val.split(',')):
                    return [cast_value(item.strip(), sub_data_type) for item in input_val.split(',')]
                elif ',' in input_val and output_type == "DICT":
                    items = {}
                    for part in input_val.split(','):
                        key, value = part.split(':', 1)
                        items[key.strip()] = cast_value(value.strip(), sub_data_type)
                    return items
                else:
                    kv_pattern = r'^(.+?)\s*[:]\s*(.+)$'
                    list_item_pattern = r'^\s*(?:-|\d+\))\s*(.+)$'
                    lines = input_val.split('\n')
                    items = {}
                    list_items = []
                    for line in lines:
                        kv_match = re.match(kv_pattern, line)
                        if kv_match:
                            key, value = kv_match.groups()
                            items[key.strip()] = cast_value(value.strip(), sub_data_type)
                        else:
                            list_item_match = re.match(list_item_pattern, line)
                            if list_item_match:
                                list_items.append(cast_value(list_item_match.group(1).strip(), sub_data_type))
                            else:
                                try:
                                    float_casted_value = cast_value(line.strip(), 'FLOAT')
                                    if float_casted_value or float_casted_value == 0.0:
                                        list_items.append(float_casted_value)
                                except ValueError:
                                    pass

                    return items if items else list_items

            elif isinstance(input_val, dict):
                return {k: cast_value(v, sub_data_type) for k, v in input_val.items()}
            elif isinstance(input_val, list):
                return [cast_value(val, sub_data_type) for val in input_val]
            else:
                return [cast_value(input_val, sub_data_type)]

        if index_or_key != "" and isinstance(input_value, (list, dict)):
            try:
                if isinstance(input_value, list):
                    input_value = input_value[int(index_or_key)]
                elif isinstance(input_value, dict):
                    if index_or_key in input_value:
                        input_value = input_value[index_or_key]
                    else:
                        raise KeyError
            except (ValueError, IndexError, KeyError, TypeError) as e:
                print(f"Error: Invalid index or key '{index_or_key}'. Defaulting to base value for {output_type}. Exception: {e}")
                return (default_values[output_type], )
        elif index_or_key == "" and output_type == "STRING":
            return (json.dumps(input_value, indent=4),)

        try:
            processed_input = process_input_value(input_value)
            if output_type == "LIST":
                output = processed_input if isinstance(processed_input, list) else list(processed_input.values())
            elif output_type == "DICT":
                output = processed_input
            else:
                print(f"Error: Unsupported type '{output_type}' for conversion. Defaulting to LIST.")
                output = processed_input if isinstance(processed_input, list) else list(processed_input.values())
        except (ValueError, TypeError) as e:
            print(f"Error: Conversion failed. Defaulting to base value.")
            output = default_values.get(output_type, [])

        return (output,)

```
