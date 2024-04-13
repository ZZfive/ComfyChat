# KSamplerProvider
## Documentation
- Class name: `KSamplerProvider`
- Category: `ImpactPack/Sampler`
- Output node: `False`

KSamplerProvider is designed to facilitate the creation of custom samplers for generative models, allowing users to specify various parameters such as seed, steps, and configuration settings to tailor the sampling process.
## Input types
### Required
- **`seed`**
    - The seed parameter ensures reproducibility of the sampling process by initializing the random number generator to a specific state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps the sampler will take, affecting the detail and quality of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration setting that influences the behavior of the sampler, allowing for fine-tuning of the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the name of the sampler to be used, enabling the selection from a predefined list of samplers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling algorithm to be applied during the sampling process, chosen from a predefined list.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Controls the level of denoising applied to the samples, affecting the clarity and sharpness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`basic_pipe`**
    - A basic pipeline configuration that includes the model and conditioning information, serving as the foundation for the sampling process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
## Output types
- **`ksampler`**
    - Comfy dtype: `KSAMPLER`
    - Returns a custom sampler instance configured according to the specified parameters.
    - Python dtype: `KSamplerWrapper`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    def doit(self, seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise)
        return (sampler, )

```
