# AnimateDiff Unload 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffUnload`
- Category: `Animate Diff 🎭🅐🅓/extras`
- Output node: `False`

The ADE_AnimateDiffUnload node is designed to revert any modifications made to the motion modules of a model, effectively restoring it to its original, unaltered state. This process involves unloading or ejecting specific parameters from the model, ensuring that it can be used in its vanilla form without the influence of previously applied AnimateDiff settings.
## Input types
### Required
- **`model`**
    - The 'model' parameter represents the model to be restored to its original state. It is crucial for specifying the target model from which the AnimateDiff modifications will be unloaded, ensuring the model can be returned to its vanilla configuration.
    - Comfy dtype: `MODEL`
    - Python dtype: `ModelPatcher`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The output is a clone of the input model, but with AnimateDiff modifications unloaded, effectively restoring the model to its original, unmodified state.
    - Python dtype: `ModelPatcher`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffUnload:
    def __init__(self) -> None:
        self.change = IsChangedHelper()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/extras"
    FUNCTION = "unload_motion_modules"

    def unload_motion_modules(self, model: ModelPatcher):
        # return model clone with ejected params
        #model = eject_params_from_model(model)
        model = get_vanilla_model_patcher(model)
        return (model.clone(),)

```
