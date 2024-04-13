# Mikey Sampler Base Only Advanced
## Documentation
- Class name: `Mikey Sampler Base Only Advanced`
- Category: `Mikey/Sampling`
- Output node: `False`

This node specializes in advanced sampling techniques for base models only, focusing on generating high-quality samples by leveraging intricate algorithms and optimizations specific to base models. It aims to enhance the output quality and efficiency of sampling processes without the need for additional refinement steps.
## Input types
### Required
- **`base_model`**
    - Specifies the base model used for sampling, serving as the foundation for generating samples.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`positive_cond_base`**
    - Defines the positive conditioning to guide the sampling process towards desired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative_cond_base`**
    - Defines the negative conditioning to steer the sampling process away from undesired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`samples`**
    - Represents the initial latent samples to be processed and refined by the node.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Specifies the variational autoencoder used in conjunction with the base model to enhance sample quality.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`add_noise`**
    - Determines whether noise should be added to the samples, enhancing diversity and potentially improving quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`denoise`**
    - Controls the level of denoising applied to the samples, affecting the clarity and sharpness of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Specifies the number of steps to be taken in the sampling process, impacting the detail and quality of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`smooth_step`**
    - Adjusts the smoothness of the transition between steps in the sampling process, affecting the gradual evolution of the samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg_1`**
    - Configures the first set of parameters for the sampling algorithm, allowing for fine-tuning of the process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cfg_2`**
    - Configures the second set of parameters for the sampling algorithm, providing additional control over the sampling characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampling algorithm to be used, influencing the approach and techniques employed in generating samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduling algorithm for managing the sampling process, affecting the progression and adaptation over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_model`**
    - Specifies the model used for upscaling the samples, enhancing their resolution and detail.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - Sets the seed for random number generation, ensuring reproducibility of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`upscale_by`**
    - Determines the factor by which the samples are upscaled, affecting their final size and resolution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`hires_denoise`**
    - Controls the level of denoising applied to the high-resolution samples, impacting their clarity and detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`hires_steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The primary output of the node, consisting of generated samples after processing through the base model and additional algorithms.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)



## Source code
```python
class MikeySamplerBaseOnlyAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        s.image_scaler = ImageScaleBy()
        s.upscale_models = folder_paths.get_filename_list("upscale_models")
        s.all_upscale_models = s.upscale_models + s.image_scaler.upscale_methods
        try:
            default_model = 'lanczos' #'4x-UltraSharp.pth' if '4x-UltraSharp.pth' in s.upscale_models else s.upscale_models[0]
            um = (s.all_upscale_models, {'default': default_model})
        except:
            um = (folder_paths.get_filename_list("upscale_models"), )
        return {"required": {"base_model": ("MODEL",),
                             "positive_cond_base": ("CONDITIONING",),
                             "negative_cond_base": ("CONDITIONING",),
                             "samples": ("LATENT",),
                             "vae": ("VAE",),
                             "add_noise": (["enable","disable"], {"default": "enable"}),
                             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "steps": ("INT", {"default": 31, "min": 1, "max": 1000}),
                             'smooth_step': ("INT", {"default": 0, "min": -1, "max": 100}),
                             "cfg_1": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                             "cfg_2": ("FLOAT", {"default": 9.5, "min": 0.1, "max": 100.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {'default': 'dpmpp_3m_sde_gpu'}),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {'default': 'exponential'}),
                             "upscale_model": um,
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "hires_denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "hires_steps": ("INT", {"default": 31, "min": 1, "max": 1000}),
                             }}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def run(self, seed, base_model, positive_cond_base, negative_cond_base,
            samples, vae, add_noise, denoise, steps, cfg_1, cfg_2, sampler_name,
            scheduler, upscale_model, upscale_by, hires_denoise, hires_steps, smooth_step):
        image_scaler = ImageScale()
        vaeencoder = VAEEncode()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        if upscale_model in image_scaler.upscale_methods:
            upscale_model = upscale_model
        else:
            upscale_model = uml.load_model(upscale_model)[0]
        iuwm = ImageUpscaleWithModel()
        # step 1 run base model low cfg
        start_step = int(steps - (steps * denoise))
        if start_step > steps // 2:
            last_step = steps - 1
        else:
            if start_step % 2 == 0:
                last_step = steps // 2 - 1
            else:
                last_step = steps // 2
        #print(f'base model start_step: {start_step}, last_step: {last_step}')
        sample1 = common_ksampler(base_model, seed, steps, cfg_1, sampler_name, scheduler,
                                  positive_cond_base, negative_cond_base, samples,
                                  start_step=start_step, last_step=last_step, force_full_denoise=False)[0]
        # step 2 run base model high cfg
        start_step = last_step + 1
        total_steps = steps + smooth_step
        sample2 = common_ksampler(base_model, seed+1, total_steps, cfg_2, sampler_name, scheduler,
                                  positive_cond_base, negative_cond_base, sample1,
                                  disable_noise=True, start_step=start_step, force_full_denoise=True)
        if upscale_by == 0:
            return sample2
        else:
            sample2 = sample2[0]
        # step 3 upscale
        pixels = vaedecoder.decode(vae, sample2)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        if isinstance(upscale_model, str):
            img = self.image_scaler.upscale(pixels, upscale_model, upscale_by)[0]
        else:
            img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
        if hires_denoise == 0:
            return (vaeencoder.encode(vae, img)[0],)
        # encode image
        latent = vaeencoder.encode(vae, img)[0]
        # step 3 run base model
        start_step = int(hires_steps - (hires_steps * hires_denoise))
        out = common_ksampler(base_model, seed, hires_steps, cfg_2, sampler_name, scheduler,
                              positive_cond_base, negative_cond_base, latent,
                              start_step=start_step, force_full_denoise=True)
        return out

```
