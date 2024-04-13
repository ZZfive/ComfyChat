# Load Face Analysis Model (mtb)
## Documentation
- Class name: `Load Face Analysis Model (mtb)`
- Category: `mtb/facetools`
- Output node: `False`

This node is designed to load a face analysis model, facilitating the analysis of facial features and characteristics within images. It supports loading different models tailored for face analysis, enhancing the flexibility and applicability of face-related operations in various contexts.
## Input types
### Required
- **`faceswap_model`**
    - Specifies the face analysis model to be loaded. The choice of model affects the analysis capabilities and performance, allowing for customization based on specific requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`face_analysis_model`**
    - Comfy dtype: `FACE_ANALYSIS_MODEL`
    - The loaded face analysis model, ready for use in analyzing facial features and characteristics within images.
    - Python dtype: `insightface.app.FaceAnalysis`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadFaceAnalysisModel:
    """Loads a face analysis model"""

    models = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "faceswap_model": (
                    ["antelopev2", "buffalo_l", "buffalo_m", "buffalo_sc"],
                    {"default": "buffalo_l"},
                ),
            },
        }

    RETURN_TYPES = ("FACE_ANALYSIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "mtb/facetools"

    def load_model(self, faceswap_model: str):
        if faceswap_model == "antelopev2":
            download_antelopev2()

        face_analyser = insightface.app.FaceAnalysis(
            name=faceswap_model,
            root=get_model_path("insightface"),
        )
        return (face_analyser,)

```
