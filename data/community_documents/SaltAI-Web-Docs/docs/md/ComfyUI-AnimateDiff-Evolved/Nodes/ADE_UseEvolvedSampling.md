# Use Evolved Sampling 🎭🅐🅓②
## Documentation
- Class name: `ADE_UseEvolvedSampling`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②`
- Output node: `False`

The ADE_UseEvolvedSampling node integrates evolved sampling techniques into the AnimateDiff framework, enhancing the generation of animations by applying advanced sampling strategies. This node is designed to leverage evolved algorithms for improved sampling efficiency and quality in the context of animation generation.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for evolved sampling, playing a crucial role in determining the quality and efficiency of the generated animations.
    - Comfy dtype: `MODEL`
    - Python dtype: `ModelPatcher`
- **`beta_schedule`**
    - Defines the beta schedule for the sampling process, influencing the progression of sampling steps and their impact on the animation quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`m_models`**
    - Optional parameter that allows for the specification of motion models, which can enhance the animation generation process by providing additional motion-related information.
    - Comfy dtype: `M_MODELS`
    - Python dtype: `MotionModelGroup`
- **`context_options`**
    - Optional parameter for setting context options, which can modify the sampling behavior based on the provided context, affecting the final animation output.
    - Comfy dtype: `CONTEXT_OPTIONS`
    - Python dtype: `ContextOptionsGroup`
- **`sample_settings`**
    - Optional parameter for defining sample settings, which can include various sampling parameters and options to fine-tune the animation generation process.
    - Comfy dtype: `SAMPLE_SETTINGS`
    - Python dtype: `SampleSettings`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the model enhanced with evolved sampling techniques, ready for use in generating animations within the AnimateDiff framework.
    - Python dtype: `ModelPatcher`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [ToBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ToBasicPipe.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - LCMScheduler



## Source code
```python
class UseEvolvedSamplingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
            },
            "optional": {
                "m_models": ("M_MODELS",),
                "context_options": ("CONTEXT_OPTIONS",),
                "sample_settings": ("SAMPLE_SETTINGS",),
                #"beta_schedule_override": ("BETA_SCHEDULE",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②"
    FUNCTION = "use_evolved_sampling"

    def use_evolved_sampling(self, model: ModelPatcher, beta_schedule: str, m_models: MotionModelGroup=None, context_options: ContextOptionsGroup=None,
                             sample_settings: SampleSettings=None, beta_schedule_override=None):
        if m_models is not None:
            m_models = m_models.clone()
            # for each motion model, confirm that it is compatible with SD model
            for motion_model in m_models.models:
                validate_model_compatibility_gen2(model=model, motion_model=motion_model)
            # create injection params
            model_name_list = [motion_model.model.mm_info.mm_name for motion_model in m_models.models]
            model_names = ",".join(model_name_list)
            # TODO: check if any apply_v2_properly is set to False
            params = InjectionParams(unlimited_area_hack=False, model_name=model_names)
        else:
            params = InjectionParams()
        # apply context options
        if context_options:
            params.set_context(context_options)
        # need to use a ModelPatcher that supports injection of motion modules into unet
        model = ModelPatcherAndInjector(model)
        model.motion_models = m_models
        model.sample_settings = sample_settings if sample_settings is not None else SampleSettings()
        model.motion_injection_params = params

        # save model_sampling from BetaSchedule as object patch
        # if autoselect, get suggested beta_schedule from motion model
        if beta_schedule == BetaSchedules.AUTOSELECT and not model.motion_models.is_empty():
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)
        
        del m_models
        return (model,)

```
