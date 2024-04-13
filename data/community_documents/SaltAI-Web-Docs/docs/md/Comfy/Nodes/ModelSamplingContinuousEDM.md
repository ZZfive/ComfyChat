# ModelSamplingContinuousEDM
## Documentation
- Class name: `ModelSamplingContinuousEDM`
- Category: `advanced/model`
- Output node: `False`

This node is designed to enhance a model's sampling capabilities by integrating continuous EDM (Energy-based Diffusion Models) sampling techniques. It allows for the dynamic adjustment of the noise levels within the model's sampling process, offering a more refined control over the generation quality and diversity.
## Input types
### Required
- **`model`**
    - The model to be enhanced with continuous EDM sampling capabilities. It serves as the foundation for applying the advanced sampling techniques.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`sampling`**
    - Specifies the type of sampling to be applied, either 'eps' for epsilon sampling or 'v_prediction' for velocity prediction, influencing the model's behavior during the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sigma_max`**
    - The maximum sigma value for noise level, allowing for upper bound control in the noise injection process during sampling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_min`**
    - The minimum sigma value for noise level, setting the lower limit for noise injection, thus affecting the model's sampling precision.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model with integrated continuous EDM sampling capabilities, ready for further use in generation tasks.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [FreeU_V2](../../Comfy/Nodes/FreeU_V2.md)



## Source code
```python
class ModelSamplingContinuousEDM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "sampling": (["v_prediction", "eps"],),
                              "sigma_max": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1000.0, "step":0.001, "round": False}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, sampling, sigma_max, sigma_min):
        m = model.clone()

        if sampling == "eps":
            sampling_type = comfy.model_sampling.EPS
        elif sampling == "v_prediction":
            sampling_type = comfy.model_sampling.V_PREDICTION

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingContinuousEDM, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_sigma_range(sigma_min, sigma_max)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )

```
