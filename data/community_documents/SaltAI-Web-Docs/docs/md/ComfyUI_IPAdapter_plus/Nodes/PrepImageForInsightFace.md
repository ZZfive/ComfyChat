# Prepare Image For InsightFace
## Documentation
- Class name: `PrepImageForInsightFace`
- Category: `ipadapter`
- Output node: `False`

The PrepImageForInsightFace node is designed to prepare images for processing with the InsightFace model. It adjusts images based on specified cropping positions, applies optional sharpening, and can add padding around the image to meet the model's requirements.
## Input types
### Required
- **`image`**
    - The input image to be processed. It is the primary subject for preparation before being fed into the InsightFace model.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`crop_position`**
    - Specifies the position for cropping the image, which can significantly impact the focus area for face detection.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sharpening`**
    - An optional parameter to apply sharpening to the image, enhancing details that may improve face detection accuracy.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pad_around`**
    - A boolean flag indicating whether to add padding around the image, which can affect the detection process by altering the image's dimensions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image, ready for further processing or analysis with the InsightFace model.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)



## Source code
```python
class PrepImageForInsightFace:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "crop_position": (["center", "top", "bottom", "left", "right"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            "pad_around": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, image, crop_position, sharpening=0.0, pad_around=True):
        if pad_around:
            padding = 30
            size = (580, 580)
        else:
            padding = 0
            size = (640, 640)
        output = prepImage(image, "LANCZOS", crop_position, size, sharpening, padding)

        return (output, )

```
