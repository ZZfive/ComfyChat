# Load IPAdapter Model
## Documentation
- Class name: `IPAdapterModelLoader`
- Category: `ipadapter`
- Output node: `False`

The IPAdapterModelLoader node is designed to load IPAdapter models from specified file paths. It ensures the model is correctly loaded and structured, particularly handling different model formats and validating the presence of essential components.
## Input types
### Required
- **`ipadapter_file`**
    - Specifies the filename of the IPAdapter model to be loaded. It is crucial for locating and loading the correct model file.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - Returns the loaded IPAdapter model, ready for further processing or application.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [IPAdapterApply](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApply.md)
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)
    - [IPAdapterApplyEncoded](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyEncoded.md)
    - ToIPAdapterPipe //Inspire
    - Reroute
    - AV_IPAdapter



## Source code
```python
class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "ipadapter"

    def load_ipadapter_model(self, ipadapter_file):
        ckpt_path = folder_paths.get_full_path("ipadapter", ipadapter_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
                    
        if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
            raise Exception("invalid IPAdapter model {}".format(ckpt_path))

        return (model,)

```
