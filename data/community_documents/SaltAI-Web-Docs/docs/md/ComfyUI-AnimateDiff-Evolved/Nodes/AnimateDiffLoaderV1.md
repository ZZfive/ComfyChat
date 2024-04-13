# AnimateDiff Loader [DEPRECATED] 🎭🅐🅓
## Documentation
- Class name: `AnimateDiffLoaderV1`
- Category: ``
- Output node: `False`

This node is designed for loading AnimateDiff models in a deprecated manner, indicating that it has been superseded by newer methods or technologies for achieving similar outcomes. It focuses on the initial setup or preparation phase of working with AnimateDiff models, specifically tailored for legacy compatibility or specific use cases that require the older loading mechanism.
## Input types
### Required
- **`model`**
    - This input specifies the AnimateDiff model to be loaded. It is crucial for determining the specific legacy model or settings to be applied, affecting the node's execution and the resulting animation capabilities.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`latents`**
    - The latents input represents the latent space vectors to be used with the AnimateDiff model. It is essential for defining the initial state or conditions under which the model operates, directly influencing the animation's characteristics.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`model_name`**
    - This input specifies the name of the AnimateDiff model to be loaded. It is important for selecting the correct legacy model or settings, affecting the node's functionality and the animations it produces.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`unlimited_area_hack`**
    - A boolean input that enables or disables the unlimited area hack, affecting how the model processes the animation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`beta_schedule`**
    - Specifies the beta schedule to be used by the AnimateDiff model. It influences the model's behavior during the animation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model output represents the loaded AnimateDiff model, ready for further processing or application.
    - Python dtype: `torch.nn.Module`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent output encapsulates the latent space vectors associated with the loaded AnimateDiff model, setting the stage for subsequent animation or transformation processes.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffLoader_Deprecated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_motion_models(),),
                "unlimited_area_hack": ("BOOLEAN", {"default": False},),
                "beta_schedule": (BetaSchedules.get_alias_list_with_first_element(BetaSchedules.SQRT_LINEAR),),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = ""
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(
        self,
        model: ModelPatcher,
        latents: Dict[str, torch.Tensor],
        model_name: str, unlimited_area_hack: bool, beta_schedule: str,
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
