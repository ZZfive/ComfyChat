# 🔧 KSampler Stochastic Variations
## Documentation
- Class name: `KSamplerVariationsStochastic+`
- Category: `essentials`
- Output node: `False`

This node is designed to generate variations of an input latent image through a stochastic process. It leverages a combination of samplers and schedulers to introduce controlled randomness and variations, aiming to produce diverse outputs from a single input. The process involves two stages of sampling with different configurations to achieve the desired variation effect.
## Input types
### Required
- **`model`**
    - The model parameter represents the neural network model used for generating variations of the latent image. It plays a crucial role in the overall operation by determining the quality and characteristics of the output.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latent_image`**
    - The latent_image parameter is the input latent representation of an image that the node will use as a basis for generating variations. It is central to the node's functionality, serving as the starting point for the stochastic variation process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`noise_seed`**
    - The noise_seed parameter is used to initialize the random noise generator. It ensures reproducibility of the variations by providing a consistent starting point for the randomness introduced during the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The steps parameter specifies the number of steps to be taken in the sampling process. It affects the extent of variation and the level of detail in the generated outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The cfg parameter controls the configuration of the generative model, influencing the strength and nature of the variations produced. It is a key factor in customizing the output according to specific requirements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler`**
    - The sampler parameter determines the specific sampling algorithm to be used in the variation process. Different samplers can introduce variations in different ways, affecting the diversity and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter selects the scheduling algorithm that controls the progression of the sampling process. It plays a significant role in managing the balance between exploration and refinement in the generation of variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - The positive parameter provides positive conditioning information to guide the generation process towards desired attributes or features in the output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative parameter provides negative conditioning information, steering the generation process away from undesired attributes or features in the output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`variation_seed`**
    - The variation_seed parameter is used to initialize the random generator for the second stage of the variation process. It ensures that the variations introduced in this stage are reproducible and consistent.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
- **`variation_strength`**
    - The variation_strength parameter controls the intensity of the variations introduced in the second stage of the process. It allows for fine-tuning the degree of change and diversity in the generated outputs.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cfg_scale`**
    - The cfg_scale parameter adjusts the configuration of the generative model for the second stage of the variation process. It influences the strength and nature of the variations produced in this stage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a modified latent image that has undergone the stochastic variation process. It represents a diverse set of variations based on the input latent image, showcasing the node's ability to generate multiple distinct outputs from a single input.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerVariationsStochastic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "model": ("MODEL",),
                    "latent_image": ("LATENT", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                    #"variation_sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, model, latent_image, noise_seed, steps, cfg, sampler, scheduler, positive, negative, variation_seed, variation_strength, cfg_scale, variation_sampler="dpmpp_2m_sde"):
        # Stage 1: composition sampler
        force_full_denoise = False # return with leftover noise = "enable"
        disable_noise = False # add noise = "enable"

        end_at_step = max(int(steps * (1-variation_strength)), 1)
        start_at_step = 0

        work_latent = latent_image.copy()
        batch_size = work_latent["samples"].shape[0]
        work_latent["samples"] = work_latent["samples"][0].unsqueeze(0)

        stage1 = common_ksampler(model, noise_seed, steps, cfg, sampler, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]
        print(stage1)
        if batch_size > 1:
            stage1["samples"] = stage1["samples"].clone().repeat(batch_size, 1, 1, 1)

        # Stage 2: variation sampler
        force_full_denoise = True
        disable_noise = True
        cfg = max(cfg * cfg_scale, 1.0)
        start_at_step = end_at_step
        end_at_step = steps

        return common_ksampler(model, variation_seed, steps, cfg, variation_sampler, scheduler, positive, negative, stage1, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

```
