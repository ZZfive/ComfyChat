# SEGM Detector (combined)
## Documentation
- Class name: `SegmDetectorCombined_v2`
- Category: `ImpactPack/Detector`
- Output node: `False`

This node is designed for combined segmentation detection, leveraging a segmentation detector to process an image and apply a threshold and dilation to the detection results. It abstracts the complexity of segmentation detection into a simple interface, enabling the generation of a mask that represents the detected segments.
## Input types
### Required
- **`segm_detector`**
    - Specifies the segmentation detector to be used for detecting segments within the image. It plays a crucial role in determining the accuracy and effectiveness of the segmentation.
    - Comfy dtype: `SEGM_DETECTOR`
    - Python dtype: `object`
- **`image`**
    - The image to be processed for segmentation detection. It serves as the primary input on which the segmentation detector operates.
    - Comfy dtype: `IMAGE`
    - Python dtype: `ndarray`
- **`threshold`**
    - A threshold value to filter the detection results based on confidence scores, affecting the sensitivity of segment detection.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`dilation`**
    - The amount of dilation applied to the detected segments, influencing the size and shape of the resulting mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask representing the detected segments within the image, after applying the specified threshold and dilation.
    - Python dtype: `ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SegmDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segm_detector": ("SEGM_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, segm_detector, image, threshold, dilation):
        mask = segm_detector.detect_combined(image, threshold, dilation)
        return (mask,)

```
