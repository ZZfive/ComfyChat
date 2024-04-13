# KSampler (Advanced/pipe)
## Documentation
- Class name: `ImpactKSamplerAdvancedBasicPipe`
- Category: `sampling`
- Output node: `False`

This node provides an advanced sampling mechanism for generating or modifying images based on a given configuration, sampler name, scheduler, and a basic pipeline setup. It encapsulates complex sampling strategies to offer enhanced control over the image generation process, leveraging advanced techniques to achieve specific visual outcomes.
## Input types
### Required
- **`basic_pipe`**
    - Represents the basic pipeline setup, providing essential components and context for the sampling operation.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`add_noise`**
    - Indicates whether noise should be added to the sampling process, affecting the texture and details of the generated images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Specifies the seed for noise generation, ensuring reproducibility of the noise patterns in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps in the sampling process, impacting the refinement and detail of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Specifies the configuration for the sampling process, influencing the behavior and quality of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Determines the specific sampling algorithm to be used, allowing for customization of the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the scheduling strategy for the sampling process, affecting how samples are generated over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_image`**
    - Provides the initial latent image to be modified or enhanced through the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`start_at_step`**
    - Specifies the starting step for the sampling process, allowing for control over the initial phase of image generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Determines the ending step for the sampling process, defining the final phase of image refinement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`return_with_leftover_noise`**
    - Indicates whether the sampling process should return the image with leftover noise, affecting the final image texture.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The basic pipeline setup, including model, clip, vae, and conditioning components.
    - Python dtype: `tuple`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation of the generated or modified image after the sampling process.
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Comfy dtype: `VAE`
    - The variational autoencoder component used in the sampling process for image generation or modification.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvancedBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     }
                }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        model, clip, vae, positive, negative = basic_pipe

        if add_noise:
            add_noise = "enable"
        else:
            add_noise = "disable"

        if return_with_leftover_noise:
            return_with_leftover_noise = "enable"
        else:
            return_with_leftover_noise = "disable"

        latent = nodes.KSamplerAdvanced().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0]
        return basic_pipe, latent, vae

```
