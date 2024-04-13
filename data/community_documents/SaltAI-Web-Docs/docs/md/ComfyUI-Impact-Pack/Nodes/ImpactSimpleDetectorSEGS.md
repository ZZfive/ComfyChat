# Simple Detector (SEGS)
## Documentation
- Class name: `ImpactSimpleDetectorSEGS`
- Category: `ImpactPack/Detector`
- Output node: `False`

The ImpactSimpleDetectorSEGS node is designed for detecting simple segmentation patterns within images. It leverages segmentation models to identify and delineate distinct segments or objects, facilitating various image analysis and manipulation tasks.
## Input types
### Required
- **`bbox_detector`**
    - Specifies the bounding box detector to be used for detection, crucial for determining the accuracy and type of segments detected.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `object`
- **`image`**
    - The input image to be processed. The quality and resolution of the image can significantly affect the detection results.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`bbox_threshold`**
    - A threshold value to determine the sensitivity of the bounding box detection. Adjusting this value can help in fine-tuning the detection results.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_dilation`**
    - Determines the extent of dilation applied to the detected bounding boxes, useful for expanding the segments to include nearby relevant pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_factor`**
    - Specifies the factor by which the detected segment's bounding box is enlarged or reduced for cropping purposes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`drop_size`**
    - Defines the minimum size for detected segments. Segments smaller than this size are dropped.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sub_threshold`**
    - A threshold value for sub-detection processes, affecting the sensitivity of further segmentation detection.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sub_dilation`**
    - Determines the extent of dilation applied in sub-detection processes, useful for adjusting segment expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sub_bbox_expansion`**
    - Specifies the expansion factor for bounding boxes in sub-detection processes, affecting the final segment size.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`sam_mask_hint_threshold`**
    - A threshold value for SAM mask hinting, influencing the mask generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`post_dilation`**
    - Determines the extent of dilation applied after all detection processes, finalizing the segment expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sam_model_opt`**
    - Optional. Specifies the SAM model options for mask generation, affecting the detection and segmentation process.
    - Comfy dtype: `SAM_MODEL`
    - Python dtype: `object`
- **`segm_detector_opt`**
    - Optional. Specifies the segmentation detector options for further refinement of detected segments.
    - Comfy dtype: `SEGM_DETECTOR`
    - Python dtype: `object`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output segments detected from the input image, including their bounding boxes, masks, and other relevant information.
    - Python dtype: `Tuple[torch.Size, List[object]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImpactMakeTileSEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactMakeTileSEGS.md)
    - [ImpactControlNetApplySEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactControlNetApplySEGS.md)
    - [SegsToCombinedMask](../../ComfyUI-Impact-Pack/Nodes/SegsToCombinedMask.md)



## Source code
```python
class SimpleDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "post_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                        "sam_model_opt": ("SAM_MODEL", ),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                      }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
               sub_threshold, sub_dilation, sub_bbox_expansion,
               sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None, segm_detector_opt=None,
               detailer_hook=None):
        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SimpleDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size, detailer_hook=detailer_hook)

        if sam_model_opt is not None:
            mask = core.make_sam_mask(sam_model_opt, segs, image, "center-1", sub_dilation,
                                      sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
            segs = core.segs_bitwise_and_mask(segs, mask)
        elif segm_detector_opt is not None:
            segm_segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size, detailer_hook=detailer_hook)
            mask = core.segs_to_combined_mask(segm_segs)
            segs = core.segs_bitwise_and_mask(segs, mask)

        segs = core.dilate_segs(segs, post_dilation)

        return (segs,)

    def doit(self, bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion,
             sam_mask_hint_threshold, post_dilation=0, sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForEach.detect(bbox_detector, image, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                            sub_threshold, sub_dilation, sub_bbox_expansion,
                                            sam_mask_hint_threshold, post_dilation=post_dilation,
                                            sam_model_opt=sam_model_opt, segm_detector_opt=segm_detector_opt)

```
