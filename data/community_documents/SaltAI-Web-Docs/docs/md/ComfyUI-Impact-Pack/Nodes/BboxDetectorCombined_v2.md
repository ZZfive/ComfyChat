# BBOX Detector (combined)
## Documentation
- Class name: `BboxDetectorCombined_v2`
- Category: `ImpactPack/Detector`
- Output node: `False`

This node is designed to perform combined bounding box detection on images, leveraging a specified bounding box detector model. It processes images to identify and delineate objects within them, applying a threshold and dilation parameter to refine the detection results. The node aims to generate precise object masks by combining the capabilities of bounding box detection and segmentation techniques.
## Input types
### Required
- **`bbox_detector`**
    - Specifies the bounding box detector model to be used for detecting objects within the image. It plays a crucial role in determining the accuracy and effectiveness of the detection process.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `str`
- **`image`**
    - The input image on which object detection is to be performed. The quality and characteristics of the image can significantly influence the detection outcomes.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`threshold`**
    - A threshold value used to filter detection results based on confidence scores. It helps in eliminating detections with low confidence, thereby improving the overall precision.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`dilation`**
    - Specifies the amount of dilation to be applied to the detected object masks. Dilation can help in expanding the masks to cover more of the object area, useful in certain detection scenarios.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask representing the detected objects within the input image. It provides a visual representation of the detection results, highlighting the areas of interest.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BboxDetectorCombined(SegmDetectorCombined):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                      }
                }

    def doit(self, bbox_detector, image, threshold, dilation):
        mask = bbox_detector.detect_combined(image, threshold, dilation)
        return (mask,)

```
