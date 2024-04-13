# SAMLoader (Impact)
## Documentation
- Class name: `SAMLoader`
- Category: `ImpactPack`
- Output node: `False`

The SAMLoader node is designed to load SAM models based on the model name and device mode specified. It supports loading different types of SAM models and ensures they are loaded onto the appropriate computing device, either GPU or CPU, based on user preference.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the SAM model to be loaded. The choice of model affects the type of SAM model instantiated and its subsequent application.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`device_mode`**
    - Determines the computing device (AUTO, Prefer GPU, CPU) on which the SAM model will be loaded. This choice influences the model's performance and resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`sam_model`**
    - Comfy dtype: `SAM_MODEL`
    - The loaded SAM model, ready for further processing or application within the system.
    - Python dtype: `SAMModel`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [ToDetailerPipe](../../ComfyUI-Impact-Pack/Nodes/ToDetailerPipe.md)
    - [SAMDetectorCombined](../../ComfyUI-Impact-Pack/Nodes/SAMDetectorCombined.md)
    - [ToDetailerPipeSDXL](../../ComfyUI-Impact-Pack/Nodes/ToDetailerPipeSDXL.md)
    - Reroute
    - [ImpactSimpleDetectorSEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactSimpleDetectorSEGS.md)
    - [ImpactSimpleDetectorSEGS_for_AD](../../ComfyUI-Impact-Pack/Nodes/ImpactSimpleDetectorSEGS_for_AD.md)
    - [GroundingDinoSAMSegment (segment anything)](../../comfyui_segment_anything/Nodes/GroundingDinoSAMSegment (segment anything).md)



## Source code
```python
class SAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = [x for x in folder_paths.get_filename_list("sams") if 'hq' not in x]
        return {
            "required": {
                "model_name": (models, ),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            }
        }

    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    def load_model(self, model_name, device_mode="auto"):
        modelname = folder_paths.get_full_path("sams", model_name)

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        size = os.path.getsize(modelname)
        sam.safe_to = core.SafeToGPU(size)

        # Unless user explicitly wants to use CPU, we use GPU
        device = comfy.model_management.get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            sam.safe_to.to_device(sam, device)

        sam.is_auto_mode = device_mode == "AUTO"

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam, )

```
