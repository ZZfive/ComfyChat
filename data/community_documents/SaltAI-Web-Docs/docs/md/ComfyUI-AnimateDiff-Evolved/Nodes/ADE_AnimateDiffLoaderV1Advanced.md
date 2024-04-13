# AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffLoaderV1Advanced`
- Category: ``
- Output node: `False`

This node is designed for advanced loading of AnimateDiff models, offering enhanced control and customization options for the loading process. It is marked as deprecated, indicating that it may no longer be the preferred method for loading these models, but it remains available for use in specific scenarios where its advanced capabilities are required.
## Input types
### Required
- **`model`**
    - Specifies the model to be loaded, serving as the core component for the AnimateDiff process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latents`**
    - Defines the latent space representations to be utilized in conjunction with the model.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`model_name`**
    - Determines the specific motion model to be loaded, allowing for targeted application of AnimateDiff functionalities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`unlimited_area_hack`**
    - A boolean flag that, when enabled, allows for bypassing certain constraints, potentially enhancing the flexibility of the AnimateDiff process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`context_length`**
    - Specifies the length of the context used in the AnimateDiff process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_stride`**
    - Defines the stride of the context, affecting how the context is sampled or processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_overlap`**
    - Indicates the overlap between consecutive contexts, which can affect the continuity and smoothness of the AnimateDiff output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_schedule`**
    - Determines the scheduling of the context, which can influence the temporal dynamics of the AnimateDiff process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`closed_loop`**
    - A boolean flag indicating whether the AnimateDiff process should operate in a closed-loop manner, potentially affecting the model's behavior.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`beta_schedule`**
    - Specifies the schedule for the beta parameter, influencing the behavior of the AnimateDiff model over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The loaded AnimateDiff model, ready for application within the specified context.
    - Python dtype: `torch.nn.Module`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent space representations associated with the loaded model, facilitating further manipulation or application.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffLoaderAdvanced_Deprecated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_motion_models(),),
                "unlimited_area_hack": ("BOOLEAN", {"default": False},),
                "context_length": ("INT", {"default": 16, "min": 0, "max": 1000}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "context_schedule": (ContextSchedules.LEGACY_UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                "beta_schedule": (BetaSchedules.get_alias_list_with_first_element(BetaSchedules.SQRT_LINEAR),),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = ""
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(self,
            model: ModelPatcher,
            latents: Dict[str, torch.Tensor],
            model_name: str, unlimited_area_hack: bool,
            context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool,
            beta_schedule: str,
        ):
        # load motion module
        motion_model = load_motion_module_gen1(model_name, model)
        # get total frames
        init_frames_len = len(latents["samples"])  # deprecated - no longer used for anything lol
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=unlimited_area_hack,
                apply_mm_groupnorm_hack=True,
                model_name=model_name,
                apply_v2_properly=False,
        )
        # set context settings
        params.set_context(
            ContextOptions(
                context_length=context_length,
                context_stride=context_stride,
                context_overlap=context_overlap,
                context_schedule=context_schedule,
                closed_loop=closed_loop,
            )
        )
        # inject for use in sampling code
        model = ModelPatcherAndInjector(model)
        model.motion_models = MotionModelGroup(motion_model)
        model.motion_injection_params = params

        # save model sampling from BetaSchedule as object patch
        # if autoselect, get suggested beta_schedule from motion model
        if beta_schedule == BetaSchedules.AUTOSELECT and not model.motion_models.is_empty():
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)

        del motion_model
        return (model, latents)

```
