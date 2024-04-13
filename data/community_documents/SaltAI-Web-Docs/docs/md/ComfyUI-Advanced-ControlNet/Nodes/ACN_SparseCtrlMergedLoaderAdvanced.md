# Load Merged SparseCtrl Model 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlMergedLoaderAdvanced`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/experimental`
- Output node: `False`

This node is designed to load a merged Sparse Control model, integrating advanced control mechanisms for enhanced manipulation and generation capabilities within the context of AI-driven creative processes. It leverages sparse control techniques to efficiently manage and apply complex control structures, facilitating the creation of nuanced and dynamic content.
## Input types
### Required
- **`sparsectrl_name`**
    - Specifies the name of the sparse control model to be loaded. This parameter is crucial for identifying and retrieving the correct model from a predefined list of available control nets.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`control_net_name`**
    - Specifies the name of the control net to be merged with the sparse control model, essential for combining the functionalities of both models for advanced content generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_motion`**
    - Determines whether motion-based control mechanisms should be utilized in the sparse control model, enhancing dynamic content generation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`motion_strength`**
    - Adjusts the intensity of motion effects within the sparse control model, allowing for fine-tuned control over the dynamism of generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`motion_scale`**
    - Scales the overall impact of motion within the sparse control model, providing a means to adjust the extent of motion-based effects on the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`sparse_method`**
    - Defines the method of sparsity to be applied in the control model, influencing how control signals are distributed and applied for content generation.
    - Comfy dtype: `SPARSE_METHOD`
    - Python dtype: `str`
- **`tk_optional`**
    - Optionally includes a timestep keyframe group to be used in conjunction with the sparse control model, enabling more precise temporal control over content generation.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `str`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - Returns the loaded sparse control model, ready for integration and use in advanced content generation processes.
    - Python dtype: `SparseCtrlAdvanced`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SparseCtrlMergedLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sparsectrl_name": (folder_paths.get_filename_list("controlnet"), ),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "use_motion": ("BOOLEAN", {"default": True}, ),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "sparse_method": ("SPARSE_METHOD", ),
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/experimental"

    def load_controlnet(self, sparsectrl_name: str, control_net_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None):
        sparsectrl_path = folder_paths.get_full_path("controlnet", sparsectrl_name)
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale, merged=True)
        # first, load normal controlnet
        controlnet = load_controlnet(controlnet_path, timestep_keyframe=tk_optional)
        # confirm that controlnet is ControlNetAdvanced
        if controlnet is None or type(controlnet) != ControlNetAdvanced:
            raise ValueError(f"controlnet_path must point to a normal ControlNet, but instead: {type(controlnet).__name__}")
        # next, load sparsectrl, making sure to load motion portion
        sparsectrl = load_sparsectrl(sparsectrl_path, timestep_keyframe=tk_optional, sparse_settings=SparseSettings.default())
        # now, combine state dicts
        new_state_dict = controlnet.control_model.state_dict()
        for key, value in sparsectrl.control_model.motion_holder.motion_wrapper.state_dict().items():
            new_state_dict[key] = value
        # now, reload sparsectrl with real settings
        sparsectrl = load_sparsectrl(sparsectrl_path, controlnet_data=new_state_dict, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return (sparsectrl,)

```
