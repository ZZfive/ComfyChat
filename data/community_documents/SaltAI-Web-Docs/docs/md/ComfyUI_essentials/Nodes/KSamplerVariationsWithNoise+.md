# 🔧 KSampler Variations with Noise Injection
## Documentation
- Class name: `KSamplerVariationsWithNoise+`
- Category: `essentials`
- Output node: `False`

This node is designed to generate variations of a given latent image by applying a two-stage sampling process with noise control. Initially, it samples variations with specified noise settings and then refines these variations under different noise conditions, allowing for a diverse range of outputs.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model to be used for sampling, playing a crucial role in determining the quality and characteristics of the generated variations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latent_image`**
    - The latent_image parameter represents the initial latent representation of an image, serving as the starting point for the generation of variations.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`main_seed`**
    - The main_seed parameter is used to initialize the random noise generator for the first stage of sampling, ensuring reproducibility of the initial noise patterns.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
- **`steps`**
    - The steps parameter determines the number of sampling steps to be performed, affecting the detail and quality of the generated variations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The cfg parameter controls the conditioning factor, influencing the diversity and fidelity of the generated variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The sampler_name parameter specifies the sampling algorithm to be used, impacting the approach to generating variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter defines the scheduling algorithm for controlling the sampling process, affecting the progression of noise reduction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - The positive parameter provides positive conditioning information, guiding the generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative parameter provides negative conditioning information, steering the generation away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`variation_strength`**
    - The variation_strength parameter adjusts the intensity of the variations introduced in the second stage of sampling, allowing for control over the diversity of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_seed`**
    - The variation_seed parameter is used to initialize the random noise generator for the second stage of sampling, ensuring reproducibility of the variation noise patterns.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a modified latent representation of the image, incorporating the generated variations.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerVariationsWithNoise:       
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "latent_image": ("LATENT", ),
                    "main_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                    #"start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    #"end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    #"return_with_leftover_noise": (["disable", "enable"], ),
                    "variation_seed": ("INT:seed", {"default": random.randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, model, latent_image, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, variation_strength, variation_seed):
        generator = torch.manual_seed(main_seed)
        batch_size, _, height, width = latent_image["samples"].shape
        base_noise = torch.randn((1, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).repeat(batch_size, 1, 1, 1).cpu()

        generator = torch.manual_seed(variation_seed)
        variation_noise = torch.randn((batch_size, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).cpu()

        slerp_noise = slerp(variation_strength, base_noise, variation_noise)

        device = comfy.model_management.get_torch_device()
        end_at_step = steps #min(steps, end_at_step)
        start_at_step = 0 #min(start_at_step, end_at_step)
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        work_latent = latent_image.copy()
        work_latent["samples"] = latent_image["samples"].clone() + slerp_noise * sigma

        force_full_denoise = True
        #if return_with_leftover_noise == "enable":
        #    force_full_denoise = False

        disable_noise = True

        return common_ksampler(model, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

```
