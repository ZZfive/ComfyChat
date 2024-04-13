# Load InsightFace
## Documentation
- Class name: `InsightFaceLoader`
- Category: `ipadapter`
- Output node: `False`

The InsightFaceLoader node is designed to load the InsightFace model, which is a facial recognition and analysis framework. It initializes the model with a specified provider for execution, preparing it for face analysis tasks.
## Input types
### Required
- **`provider`**
    - Specifies the execution provider for the InsightFace model, such as CPU, CUDA, or ROCM. This choice determines the computational backend that will be used for model operations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`insightface`**
    - Comfy dtype: `INSIGHTFACE`
    - Returns an initialized InsightFace model ready for facial analysis tasks.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)



## Source code
```python
class InsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load_insight_face"
    CATEGORY = "ipadapter"

    def load_insight_face(self, provider):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise Exception(e)
        
        from insightface.utils import face_align
        global insightface_face_align
        insightface_face_align = face_align

        model = FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

```
