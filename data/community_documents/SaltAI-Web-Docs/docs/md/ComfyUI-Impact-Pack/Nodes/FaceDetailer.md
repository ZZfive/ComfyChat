# FaceDetailer
## Documentation
- Class name: `FaceDetailer`
- Category: `ImpactPack/Simple`
- Output node: `False`

The FaceDetailer node is designed to enhance the details of faces in images, utilizing a combination of models and techniques to refine facial features and improve overall image quality. It is specifically tailored for single images and warns against its use for video detailing, directing users to alternative nodes for such purposes.
## Input types
### Required
- **`image`**
    - The input image to be enhanced. The node focuses on refining facial details within this image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`model`**
    - The primary model used for enhancing facial details in the image.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - A CLIP model used for guiding the enhancement process based on textual descriptions.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - A Variational Autoencoder (VAE) model used for generating or refining facial features within the image.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`guide_size`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`guide_size_for`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`max_size`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`cfg`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sampler_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`scheduler`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`positive`**
    - Positive textual prompts that guide the enhancement process towards desired facial features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list[str]`
- **`negative`**
    - Negative textual prompts used to steer the enhancement process away from undesired facial features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list[str]`
- **`denoise`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`noise_mask`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`force_inpaint`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`bbox_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`bbox_dilation`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`bbox_crop_factor`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_detection_hint`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`sam_dilation`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`sam_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_bbox_expansion`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`sam_mask_hint_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_mask_hint_use_negative`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`drop_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`bbox_detector`**
    - A bounding box detector used to identify and focus on facial regions within the image.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `torch.nn.Module`
- **`wildcard`**
    - Optional wildcard prompts that introduce variability into the enhancement process.
    - Comfy dtype: `STRING`
    - Python dtype: `list[str]`
- **`cycle`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`sam_model_opt`**
    - Optional SAM model configurations used to refine the selection and enhancement of facial features.
    - Comfy dtype: `SAM_MODEL`
    - Python dtype: `dict`
- **`segm_detector_opt`**
    - A segmentation model used to delineate facial features more precisely for targeted enhancement.
    - Comfy dtype: `SEGM_DETECTOR`
    - Python dtype: `torch.nn.Module`
- **`detailer_hook`**
    - Custom hooks or callbacks that allow for additional processing or modification during the enhancement process.
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `callable`
- **`inpaint_model`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`noise_mask_feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced image with refined facial details.
    - Python dtype: `torch.Tensor`
- **`cropped_refined`**
    - Comfy dtype: `IMAGE`
    - A list of cropped regions from the image that have been enhanced.
    - Python dtype: `list[torch.Tensor]`
- **`cropped_enhanced_alpha`**
    - Comfy dtype: `IMAGE`
    - A list of alpha masks corresponding to the cropped enhanced regions, indicating the level of enhancement.
    - Python dtype: `list[torch.Tensor]`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask indicating the regions of the image that were enhanced.
    - Python dtype: `torch.Tensor`
- **`detailer_pipe`**
    - Comfy dtype: `DETAILER_PIPE`
    - A tuple containing the models and configurations used during the enhancement process.
    - Python dtype: `tuple`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - A list of images processed through the CNet model, providing additional detail enhancement.
    - Python dtype: `list[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [FaceDetailerPipe](../../ComfyUI-Impact-Pack/Nodes/FaceDetailerPipe.md)
    - [MaskToImage](../../Comfy/Nodes/MaskToImage.md)
    - Reroute
    - [SaveImage](../../Comfy/Nodes/SaveImage.md)
    - [PlaySound|pysssss](../../ComfyUI-Custom-Scripts/Nodes/PlaySound|pysssss.md)
    - [InvertMask](../../Comfy/Nodes/InvertMask.md)
    - [ImageInvert](../../Comfy/Nodes/ImageInvert.md)
    - [ImageCompositeMasked](../../Comfy/Nodes/ImageCompositeMasked.md)
    - [ImageBlend](../../Comfy/Nodes/ImageBlend.md)



## Source code
```python
class FaceDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                     "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                     "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                     "bbox_detector": ("BBOX_DETECTOR", ),
                     "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                    "segm_detector_opt": ("SEGM_DETECTOR", ),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Simple"

    @staticmethod
    def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector, segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, cycle=1,
                     inpaint_model=False, noise_mask_feather=0):

        # make default prompt as 'face' if empty prompt for CLIPSeg
        bbox_detector.setAux('face')
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size, detailer_hook=detailer_hook)
        bbox_detector.setAux(None)

        # bbox + sam combination
        if sam_model_opt is not None:
            sam_mask = core.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                          sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                          sam_mask_hint_use_negative, )
            segs = core.segs_bitwise_and_mask(segs, sam_mask)

        elif segm_detector is not None:
            segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

            if (hasattr(segm_detector, 'override_bbox_by_segm') and segm_detector.override_bbox_by_segm and
                    not (detailer_hook is not None and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                segs = segm_segs
            else:
                segm_mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, segm_mask)

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                                          sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                          force_inpaint, wildcard_opt, detailer_hook,
                                          refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                          refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                          refiner_negative=refiner_negative,
                                          cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []

        # Mask Generator
        mask = core.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def doit(self, image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector, wildcard, cycle=1,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0):

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []

        if len(image) > 1:
            print(f"[Impact Pack] WARN: FaceDetailer is not a node designed for video detailing. If you intend to perform video detailing, please use Detailer For AnimateDiff.")

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = FaceDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg, sampler_name, scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images

```
