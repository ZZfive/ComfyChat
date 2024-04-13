# Apply IPAdapter FaceID
## Documentation
- Class name: `IPAdapterApplyFaceID`
- Category: `ipadapter`
- Output node: `False`

This node specializes in applying facial identification enhancements to images using IPAdapter technology. It leverages InsightFace for face detection and embedding, and integrates with CLIP vision models to enhance image processing with facial recognition capabilities.
## Input types
### Required
- **`ipadapter`**
    - The IPAdapter instance to be used for processing the image. It's crucial for defining the transformation and enhancement parameters specific to the image.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `IPADAPTER`
- **`clip_vision`**
    - A CLIP vision model instance used for enhancing the image processing with visual understanding capabilities.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `CLIP_VISION`
- **`insightface`**
    - An InsightFace model instance used for detecting and embedding faces within the image. Essential for facial identification tasks.
    - Comfy dtype: `INSIGHTFACE`
    - Python dtype: `INSIGHTFACE`
- **`image`**
    - The image to be processed and enhanced with facial identification features.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`model`**
    - The model used for applying the IPAdapter transformations and enhancements to the image.
    - Comfy dtype: `MODEL`
    - Python dtype: `MODEL`
- **`weight`**
    - A float value representing the weight of the transformation effect on the image. It influences the intensity of the applied enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - A float value indicating the amount of noise to be added to the image. This can affect the visual quality and the effectiveness of facial identification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the type of weighting to be applied during the transformation. Options include 'original', 'linear', and 'channel penalty', each affecting the enhancement differently.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - A float value indicating the starting point of the transformation effect. It determines the initial intensity of the applied enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - A float value indicating the ending point of the transformation effect. It determines the final intensity of the applied enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`faceid_v2`**
    - A boolean indicating whether to use the version 2 of the facial identification enhancements. This can alter the processing and results significantly.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`weight_v2`**
    - A float value representing the weight of the version 2 transformation effect on the image. It influences the intensity of the applied enhancements when faceid_v2 is true.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - A boolean indicating whether to unfold the batch during processing. This can affect the performance and results of the facial identification enhancements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`attn_mask`**
    - An optional mask to be applied during the transformation. It can direct the focus of the enhancements and affect the outcome.
    - Comfy dtype: `MASK`
    - Python dtype: `MASK`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model output represents the enhanced image after applying facial identification and other transformations. It encapsulates the modifications made to the original image, incorporating facial recognition capabilities.
    - Python dtype: `MODEL`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - KSampler //Inspire
    - [FreeU_V2](../../Comfy/Nodes/FreeU_V2.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)
    - [IPAdapterApply](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApply.md)
    - ToIPAdapterPipe //Inspire
    - [ToDetailerPipe](../../ComfyUI-Impact-Pack/Nodes/ToDetailerPipe.md)
    - [CR Apply LoRA Stack](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Apply LoRA Stack.md)
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)



## Source code
```python
class IPAdapterApplyFaceID(IPAdapterApply):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
                "image": ("IMAGE",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "faceid_v2": ("BOOLEAN", { "default": False }),
                "weight_v2": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "unfold_batch": ("BOOLEAN", { "default": False }),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

```
