# MaskDetailer (pipe)
## Documentation
- Class name: `MaskDetailerPipe`
- Category: `ImpactPack/Detailer`
- Output node: `False`

The MaskDetailerPipe node is designed to enhance and refine mask details within images, leveraging advanced processing techniques to improve the quality and precision of masks. It focuses on optimizing mask outlines and features for better visual clarity and accuracy.
## Input types
### Required
- **`image`**
    - The input image to which mask detailing will be applied, serving as the base for enhancement operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask to be detailed and refined, targeting improvements in its outlines and features.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`basic_pipe`**
    - Configuration details for the basic processing pipeline, influencing the initial stages of mask detailing.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`guide_size`**
    - Specifies the size of the guide used in the detailing process, affecting the precision and scale of mask enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`max_size`**
    - Defines the maximum size limit for the detailing operation, ensuring the process adheres to performance constraints.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`mask_mode`**
    - Indicates the mode of mask processing, such as binary or grayscale, influencing how the mask is interpreted and refined.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
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
- **`denoise`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`crop_factor`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`drop_size`**
    - unknown
    - Comfy dtype: `INT`
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
- **`detailer_hook`**
    - unknown
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `unknown`
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
    - Outputs the image with enhanced and refined mask details, showcasing improved clarity and precision.
    - Python dtype: `torch.Tensor`
- **`cropped_refined`**
    - Comfy dtype: `IMAGE`
    - Provides the cropped and refined segments of the mask, highlighting the detailed areas.
    - Python dtype: `list[torch.Tensor]`
- **`cropped_enhanced_alpha`**
    - Comfy dtype: `IMAGE`
    - Outputs the alpha channel of the cropped and enhanced mask segments, offering additional detail on mask transparency and edges.
    - Python dtype: `list[torch.Tensor]`
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - Returns the configuration of the basic processing pipeline after the detailing operations.
    - Python dtype: `tuple`
- **`refiner_basic_pipe_opt`**
    - Comfy dtype: `BASIC_PIPE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "basic_pipe": ("BASIC_PIPE",),

                    "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "mask bbox", "label_off": "crop region"}),
                    "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),

                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),

                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                    "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),

                    "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                   },
                "optional": {
                    "refiner_basic_pipe_opt": ("BASIC_PIPE", ),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                   }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BASIC_PIPE", "BASIC_PIPE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "basic_pipe", "refiner_basic_pipe_opt")
    OUTPUT_IS_LIST = (False, True, True, False, False)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             refiner_basic_pipe_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: MaskDetailer does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        model, clip, vae, positive, negative = basic_pipe

        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        # create segs
        if mask is not None:
            mask = make_2d_mask(mask)
            segs = core.mask_to_segs(mask, False, crop_factor, False, drop_size)
        else:
            segs = ((image.shape[1], image.shape[2]), [])

        enhanced_img_batch = None
        cropped_enhanced_list = []
        cropped_enhanced_alpha_list = []

        for i in range(batch_size):
            if mask is not None:
                enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, _, _ = \
                    DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed+i, steps,
                                              cfg, sampler_name, scheduler, positive, negative, denoise, feather, mask_mode,
                                              force_inpaint=True, wildcard_opt=None, detailer_hook=detailer_hook,
                                              refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip,
                                              refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                              cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather)
            else:
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha = image, [], []

            if enhanced_img_batch is None:
                enhanced_img_batch = enhanced_img
            else:
                enhanced_img_batch = torch.cat((enhanced_img_batch, enhanced_img), dim=0)

            cropped_enhanced_list += cropped_enhanced
            cropped_enhanced_alpha_list += cropped_enhanced_alpha

        # set fallback image
        if len(cropped_enhanced_list) == 0:
            cropped_enhanced_list = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha_list) == 0:
            cropped_enhanced_alpha_list = [empty_pil_tensor()]

        return enhanced_img_batch, cropped_enhanced_list, cropped_enhanced_alpha_list, basic_pipe, refiner_basic_pipe_opt

```
