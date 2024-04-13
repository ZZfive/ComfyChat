# Switch (Any)
## Documentation
- Class name: `ImpactSwitch`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactSwitch node is designed to dynamically select between multiple inputs based on a specified condition or index. It serves as a control mechanism within a workflow, allowing for the conditional routing of data streams based on runtime decisions.
## Input types
### Required
- **`select`**
    - Determines which input to select and pass as output. The choice is based on an index or condition evaluated at runtime, making the node's operation dynamic and adaptable to varying scenarios.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sel_mode`**
    - Specifies the mode of selection, enabling further customization of how the 'select' input is interpreted and applied within the node's logic.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`input1`**
    - An optional input that can be selected based on the 'select' parameter. It provides additional flexibility in the node's operation by allowing for more inputs to be conditionally routed.
    - Comfy dtype: `*`
    - Python dtype: `object`
## Output types
- **`selected_value`**
    - Comfy dtype: `*`
    - The output value selected based on the 'select' parameter, demonstrating the node's ability to conditionally route data.
    - Python dtype: `object`
- **`selected_label`**
    - Comfy dtype: `STRING`
    - A label associated with the selected output, providing context or additional information about the selected data.
    - Python dtype: `str`
- **`selected_index`**
    - Comfy dtype: `INT`
    - The index of the selected input, offering insight into which condition or option was chosen during the node's execution.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)



## Source code
```python
class GeneralSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "select": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
                    "sel_mode": ("BOOLEAN", {"default": True, "label_on": "select_on_prompt", "label_off": "select_on_execution", "forceInput": False}),
                    },
                "optional": {
                    "input1": (any_typ,),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    RETURN_TYPES = (any_typ, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"

        selected_label = input_name
        node_id = kwargs['unique_id']
        nodelist = kwargs['extra_pnginfo']['workflow']['nodes']
        for node in nodelist:
            if str(node['id']) == node_id:
                inputs = node['inputs']

                for slot in inputs:
                    if slot['name'] == input_name and 'label' in slot:
                        selected_label = slot['label']

                break

        if input_name in kwargs:
            return (kwargs[input_name], selected_label, selected_index)
        else:
            print(f"ImpactSwitch: invalid select index (ignored)")
            return (None, "", selected_index)

```
