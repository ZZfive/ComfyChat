# Prepare Image For Clip Vision
## Documentation
- Class name: `PrepImageForClipVision`
- Category: `ipadapter`
- Output node: `False`

The PrepImageForClipVision node is designed to prepare images for processing by CLIP vision models. It adjusts images according to specified parameters such as interpolation, crop position, and sharpening to ensure they are in the optimal format for feature extraction by CLIP models.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image to be processed. It is crucial for the node's operation as it is the subject of the preparation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`interpolation`**
    - The 'interpolation' parameter specifies the method used to resize the image, affecting the quality and the way pixels are interpolated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`crop_position`**
    - The 'crop_position' parameter determines the area of the image to focus on or how the image is cropped, influencing the portion of the image that will be analyzed by the CLIP model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sharpening`**
    - The 'sharpening' parameter adjusts the sharpness of the image, enhancing edge definition and detail visibility, which can affect the model's ability to extract features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a processed image, optimized for feature extraction by CLIP vision models, ensuring that the image is in the best possible state for analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - IDGenerationNode
    - [IPAdapterApply](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApply.md)
    - [AIO_Preprocessor](../../comfyui_controlnet_aux/Nodes/AIO_Preprocessor.md)
    - [ImageBatch](../../Comfy/Nodes/ImageBatch.md)
    - [CLIPVisionEncode](../../Comfy/Nodes/CLIPVisionEncode.md)
    - [PrepImageForInsightFace](../../ComfyUI_IPAdapter_plus/Nodes/PrepImageForInsightFace.md)
    - SetNode
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [IPAdapterEncoder](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterEncoder.md)



## Source code
```python
class PrepImageForClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
        size = (224, 224)
        output = prepImage(image, interpolation, crop_position, size, sharpening, 0)
        return (output, )

```
