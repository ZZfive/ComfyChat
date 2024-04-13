# KSampler (pipe)
## Documentation
- Class name: `ImpactKSamplerBasicPipe`
- Category: `sampling`
- Output node: `False`

This node is designed for sampling operations within a basic pipeline, utilizing a variety of samplers and schedulers to generate or modify latent images. It abstracts the complexity of sampling algorithms, providing a streamlined interface for applying advanced sampling techniques to basic pipeline components.
## Input types
### Required
- **`basic_pipe`**
    - Represents the basic pipeline components, including models and configurations, essential for the sampling process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`seed`**
    - Determines the starting point for random number generation, ensuring reproducibility of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Specifies the number of steps to perform in the sampling process, affecting the detail and quality of the generated latent image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the configuration of the sampling process, influencing the behavior and outcomes of the sampler.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampler to use from a predefined list of available samplers, tailoring the sampling process to specific needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler to manage the sampling steps, optimizing the process for efficiency and effectiveness.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_image`**
    - The initial latent image to be modified or enhanced through the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`denoise`**
    - Adjusts the level of denoising applied to the latent image, fine-tuning the clarity and quality of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - Returns the basic pipeline components, including any modifications made during the sampling process.
    - Python dtype: `tuple`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The resulting latent image after the sampling process, ready for further processing or visualization.
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE component of the basic pipeline, potentially utilized or modified during the sampling process.
    - Python dtype: `VAE`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)
    - Reroute
    - [ImpactKSamplerBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ImpactKSamplerBasicPipe.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)



## Source code
```python
class KSamplerBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise=1.0):
        model, clip, vae, positive, negative = basic_pipe
        latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
        return basic_pipe, latent, vae

```
