# FreeInit Iteration Options 🎭🅐🅓
## Documentation
- Class name: `ADE_IterationOptsFreeInit`
- Category: `Animate Diff 🎭🅐🅓/iteration opts`
- Output node: `False`

This node is designed to create customized iteration options for the AnimateDiff process, allowing users to specify various parameters such as the number of iterations, batch offsets, and seed offsets. It facilitates the generation of iteration configurations that can be tailored to different animation requirements, enhancing the flexibility and control over the animation generation process.
## Input types
### Required
- **`iterations`**
    - Specifies the number of iterations to be performed. This parameter is crucial for defining the length and detail of the animation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`filter`**
    - Defines the filter to be applied during the iteration process, affecting the visual characteristics of the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `FreeInitFilter`
- **`d_s`**
    - Specifies the start distance for the filter effect, influencing the initial impact of the filter on the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`d_t`**
    - Determines the end distance for the filter effect, affecting the final appearance of the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`n_butterworth`**
    - Sets the order of the Butterworth filter, controlling the smoothness and sharpness of the filter effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_step`**
    - Specifies the step size for the sigma parameter in the filtering process, affecting the granularity of the filter effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`apply_to_1st_iter`**
    - Indicates whether the filter should be applied to the first iteration, affecting the starting point of the animation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`init_type`**
    - Defines the initialization type for the iteration process, influencing the starting conditions of the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `FreeInitOptions`
### Optional
- **`iter_batch_offset`**
    - Determines the offset for batch processing during iterations. This can be used to adjust the starting point for batch processing, affecting how animation frames are generated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`iter_seed_offset`**
    - Sets the seed offset for iterations, influencing the randomness and variation in the animation process. This can be used to achieve different visual effects or to replicate specific animation outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`iteration_opts`**
    - Comfy dtype: `ITERATION_OPTS`
    - Provides the customized iteration options based on the specified parameters. This output is essential for configuring the AnimateDiff process according to user-defined settings.
    - Python dtype: `IterationOptions`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FreeInitOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iterations": ("INT", {"default": 2, "min": 1}),
                "filter": (FreeInitFilter.LIST,),
                "d_s": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "d_t": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "n_butterworth": ("INT", {"default": 4, "min": 1, "max": 100},),
                "sigma_step": ("INT", {"default": 999, "min": 1, "max": 999}),
                "apply_to_1st_iter": ("BOOLEAN", {"default": False}),
                "init_type": (FreeInitOptions.LIST,)
            },
            "optional": {
                "iter_batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "iter_seed_offset": ("INT", {"default": 1, "min": BIGMIN, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("ITERATION_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/iteration opts"
    FUNCTION = "create_iter_opts"

    def create_iter_opts(self, iterations: int, filter: str, d_s: float, d_t: float, n_butterworth: int,
                         sigma_step: int, apply_to_1st_iter: bool, init_type: str,
                         iter_batch_offset: int=0, iter_seed_offset: int=1):
        # init_type does nothing for now, not until I add more methods of applying low+high freq noise
        iter_opts = FreeInitOptions(iterations=iterations, step=sigma_step, apply_to_1st_iter=apply_to_1st_iter,
                                    filter=filter, d_s=d_s, d_t=d_t, n=n_butterworth, init_type=init_type,
                                    iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return (iter_opts,)

```
