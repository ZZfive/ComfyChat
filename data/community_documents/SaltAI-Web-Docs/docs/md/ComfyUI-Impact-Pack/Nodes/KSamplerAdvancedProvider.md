# KSamplerAdvancedProvider
## Documentation
- Class name: `KSamplerAdvancedProvider`
- Category: `ImpactPack/Sampler`
- Output node: `False`

This node provides an advanced KSampler configuration, allowing for the customization of sampling behavior through various parameters. It is designed to enhance the flexibility and control over the sampling process, catering to specific needs and preferences.
## Input types
### Required
- **`cfg`**
    - Specifies the configuration value for the sampler, influencing its behavior and performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Determines the specific sampler to be used, chosen from a predefined set of samplers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Selects the scheduling algorithm to be used, chosen from a predefined set of schedulers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`basic_pipe`**
    - Provides the basic pipeline components required for the sampling process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
## Output types
- **`ksampler_advanced`**
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Returns an advanced KSampler instance, configured according to the provided parameters.
    - Python dtype: `KSamplerAdvancedWrapper`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvancedProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                }

    RETURN_TYPES = ("KSAMPLER_ADVANCED",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, cfg, sampler_name, scheduler, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative)
        return (sampler, )

```
