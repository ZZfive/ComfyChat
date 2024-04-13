# Load Face Swap Model (mtb)
## Documentation
- Class name: `Load Face Swap Model (mtb)`
- Category: `mtb/facetools`
- Output node: `False`

The LoadFaceSwapModel node is designed to load a faceswap model from a specified path. It supports loading models with specific file extensions and is essential for initializing the faceswap process by ensuring the required model is available and correctly loaded.
## Input types
### Required
- **`faceswap_model`**
    - Specifies the name of the faceswap model to be loaded. This parameter is crucial for identifying and loading the correct model file for the faceswap operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`faceswap_model`**
    - Comfy dtype: `FACESWAP_MODEL`
    - Represents the loaded faceswap model, ready for use in faceswap operations.
    - Python dtype: `Tuple[INSwapper]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadFaceSwapModel:
    """Loads a faceswap model"""

    @staticmethod
    def get_models() -> List[Path]:
        models_path = get_model_path("insightface").iterdir()
        return [x for x in models_path if x.suffix in [".onnx", ".pth"]]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceswap_model": (
                    [x.name for x in cls.get_models()],
                    {"default": "None"},
                ),
            },
        }

    RETURN_TYPES = ("FACESWAP_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "mtb/facetools"

    def load_model(self, faceswap_model: str):
        model_path = get_model_path("insightface", faceswap_model)
        if not model_path or not model_path.exists():
            raise ModelNotFound(f"{faceswap_model} ({model_path})")

        log.info(f"Loading model {model_path}")
        return (
            INSwapper(
                model_path,
                onnxruntime.InferenceSession(
                    path_or_bytes=model_path,
                    providers=onnxruntime.get_available_providers(),
                ),
            ),
        )

```
