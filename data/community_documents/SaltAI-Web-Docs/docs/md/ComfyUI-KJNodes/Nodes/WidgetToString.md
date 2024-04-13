# WidgetToString
## Documentation
- Class name: `WidgetToString`
- Category: `KJNodes`
- Output node: `False`

The WidgetToString node is designed to extract and convert widget values to a string format based on specified conditions. It allows for selective retrieval of widget data within a workflow, enabling the customization of output based on user-defined parameters.
## Input types
### Required
- **`id`**
    - Specifies the unique identifier of the widget whose value is to be retrieved. It is crucial for pinpointing the exact widget within a workflow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`widget_name`**
    - Defines the name of the widget whose value is sought. This parameter is essential for identifying the specific widget within a node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`return_all`**
    - A boolean flag that determines whether to return values of all widgets or just the specified widget. It affects the breadth of data retrieved from the workflow.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns the widget values in a string format, either as a single value or a concatenated list of values, depending on the input parameters.
    - Python dtype: `Tuple[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WidgetToString:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id": ("INT", {"default": 0}),
                "widget_name": ("STRING", {"multiline": False}),
                "return_all": ("BOOLEAN", {"default": False}),
            },
            
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_widget_value"
    CATEGORY = "KJNodes"

    def get_widget_value(self, id, widget_name, extra_pnginfo, prompt, return_all=False):
        workflow = extra_pnginfo["workflow"]
        results = []
        for node in workflow["nodes"]:
            node_id = node["id"]

            if node_id != id:
                continue

            values = prompt[str(node_id)]
            if "inputs" in values:
                if return_all:
                    results.append(', '.join(f'{k}: {str(v)}' for k, v in values["inputs"].items()))
                elif widget_name in values["inputs"]:
                    v = str(values["inputs"][widget_name])  # Convert to string here
                    return (v, )
                else:
                    raise NameError(f"Widget not found: {id}.{widget_name}")
        if not results:
            raise NameError(f"Node not found: {id}")
        return (', '.join(results).strip(', '), )

```
