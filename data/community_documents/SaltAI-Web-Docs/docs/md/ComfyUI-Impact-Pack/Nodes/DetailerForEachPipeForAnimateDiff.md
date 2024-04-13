# Detailer For AnimateDiff (SEGS/pipe)
## Documentation
- Class name: `DetailerForEachPipeForAnimateDiff`
- Category: `ImpactPack/Detailer`
- Output node: `False`

This node is designed to detail each pipe specifically for the purpose of animation difference analysis within the SEGS/pipe context. It focuses on enhancing the granularity and precision of animation frames by applying detailed adjustments and analysis to each pipeline stage, thereby improving the overall animation quality and consistency.
## Input types
### Required
- **`image_frames`**
    - The sequence of image frames to be detailed for animation difference analysis. It serves as the primary input for the detailing process, influencing the enhancement of animation quality.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`segs`**
    - Segmentation information for the image frames, used to guide the detailing process by specifying areas of interest or importance.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[torch.Tensor, List[torch.Tensor]]`
- **`guide_size`**
    - Specifies the target size for guiding the detailing process, affecting the scale and precision of the adjustments made.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Determines the specific application of the guide size within the detailing process, influencing how adjustments are scaled.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`max_size`**
    - The maximum allowable size for the detailed image frames, ensuring that the detailing process does not exceed computational or quality constraints.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility and consistency in the detailing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to be taken in the detailing process, directly influencing the depth and thoroughness of the adjustments made.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the detailing process, allowing for customization and optimization of the detailing operations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampling method to be used in the detailing process, affecting the quality and characteristics of the detailed image frames.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling strategy for the detailing process, influencing the order and timing of detailing operations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - A boolean flag indicating whether denoising operations should be applied during the detailing process, affecting the clarity and quality of the detailed image frames.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`basic_pipe`**
    - The basic pipeline configuration to be used as a foundation for the detailing process, influencing the overall approach and capabilities of the detailing operations.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]`
- **`refiner_ratio`**
    - An optional ratio for refining the detailing process, allowing for fine-tuning of the adjustments made.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Optional[float]`
### Optional
- **`detailer_hook`**
    - An optional hook for custom detailing operations, providing flexibility and extensibility to the detailing process.
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `Optional[Callable]`
- **`refiner_basic_pipe_opt`**
    - An optional refined basic pipeline configuration, offering an alternative approach to the detailing process for enhanced results.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced image frames resulting from the detailing process, showcasing improved animation quality and consistency.
    - Python dtype: `List[torch.Tensor]`
- **`segs`**
    - Comfy dtype: `SEGS`
    - The updated segmentation information after the detailing process, reflecting adjustments made to areas of interest.
    - Python dtype: `Tuple[torch.Tensor, List[torch.Tensor]]`
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The basic pipeline configuration used during the detailing process, potentially modified based on detailing operations.
    - Python dtype: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)



## Source code
```python
class DetailerForEachPipeForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "image_frames": ("IMAGE", ),
                      "segs": ("SEGS", ),
                      "guide_size": ("FLOAT", {"default": 384, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                      "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                      "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                      "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                      "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                      "basic_pipe": ("BASIC_PIPE", ),
                      "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     },
                "optional": {
                     "detailer_hook": ("DETAILER_HOOK",),
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     # "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     # "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                    }
                }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE")
    RETURN_NAMES = ("image", "segs", "basic_pipe")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def doit(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, basic_pipe, refiner_ratio=None, detailer_hook=None, refiner_basic_pipe_opt=None,
             inpaint_model=False, noise_mask_feather=0):

        enhanced_segs = []
        for sub_seg in segs[1]:
            single_seg = segs[0], [sub_seg]
            enhanced_seg = SEGSDetailerForAnimateDiff().do_detail(image_frames, single_seg, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                                                                  denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt, inpaint_model, noise_mask_feather)

            image_frames = SEGSPaste.doit(image_frames, enhanced_seg, feather, alpha=255)[0]
            enhanced_segs += enhanced_seg[1]

        new_segs = segs[0], enhanced_segs
        return image_frames, new_segs, basic_pipe

```
