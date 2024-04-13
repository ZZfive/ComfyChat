# SEGSDetailer
## Documentation
- Class name: `SEGSDetailer`
- Category: `ImpactPack/Detailer`
- Output node: `False`

The SEGSDetailer node is designed to enhance the detail and clarity of segmentation results. It operates by applying specific detailing hooks to the segmentation outputs, refining their accuracy and relevance for further processing or analysis.
## Input types
### Required
- **`image`**
    - The input image or images to be detailed. This is the primary data upon which the detailing process is applied, affecting the visual quality and clarity of the output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor] or torch.Tensor`
- **`segs`**
    - The segmentation results to be enhanced. The detailing process aims to refine these results for improved accuracy and visual quality.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[torch.Size, List[custom segmentation element type]]`
- **`guide_size`**
    - Determines the size of the guide used in the detailing process, influencing the level of detail achieved in the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Specifies the guide size for a particular operation within the detailing process, affecting the detail level and clarity of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `int`
- **`max_size`**
    - The maximum size limit for the detailing process, which can impact the overall quality and detail of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the detailing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to execute in the detailing process, directly influencing the depth of detail enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the detailing process, affecting various aspects of execution and output quality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Dict[str, any]`
- **`sampler_name`**
    - Specifies the sampler to be used in the detailing process, impacting the method of detail enhancement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling strategy for the detailing process, influencing the order and manner in which details are enhanced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - A boolean flag indicating whether to apply denoising in the detailing process, affecting the clarity of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`noise_mask`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`force_inpaint`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`basic_pipe`**
    - unknown
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `unknown`
- **`refiner_ratio`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`batch_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`cycle`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`refiner_basic_pipe_opt`**
    - unknown
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `unknown`
- **`inpaint_model`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`noise_mask_feather`**
    - Specifies the feathering level for the noise mask, affecting the smoothness of transitions in the detailed output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The enhanced segmentation results, with improved detail and clarity, ready for further processing or analysis.
    - Python dtype: `Tuple[torch.Size, List[custom segmentation element type]]`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SEGSDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "basic_pipe": ("BASIC_PIPE",),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                     }
                }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "cnet_images")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
                  refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image.shape)

        new_segs = []
        cnet_pil_list = []

        for i in range(batch_size):
            seed += 1
            for seg in segs[1]:
                cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                                  else crop_ndarray4(image.numpy(), seg.crop_region)
                cropped_image = to_tensor(cropped_image)

                is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
                if is_mask_all_zeros:
                    print(f"Detailer: segment skip [empty mask]")
                    new_segs.append(seg)
                    continue

                if noise_mask:
                    cropped_mask = seg.cropped_mask
                else:
                    cropped_mask = None

                enhanced_image, cnet_pils = core.enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                positive, negative, denoise, cropped_mask, force_inpaint,
                                                                refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                                                control_net_wrapper=seg.control_net_wrapper, cycle=cycle,
                                                                inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

                if cnet_pils is not None:
                    cnet_pil_list.extend(cnet_pils)

                if enhanced_image is None:
                    new_cropped_image = cropped_image
                else:
                    new_cropped_image = enhanced_image

                new_seg = SEG(to_numpy(new_cropped_image), seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
                new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_pil_list

    def doit(self, image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
             refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SEGSDetailer does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        segs, cnet_pil_list = SEGSDetailer.do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                     scheduler, denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio, batch_size, cycle=cycle,
                                                     refiner_basic_pipe_opt=refiner_basic_pipe_opt,
                                                     inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)

        # set fallback image
        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return (segs, cnet_pil_list)

```
