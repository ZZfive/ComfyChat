# SamplerCustom
## Documentation
- Class name: `SamplerCustom`
- Category: `sampling/custom_sampling`
- Output node: `False`

The SamplerCustom node is designed to provide a flexible and customizable sampling mechanism for various applications. It enables users to select and configure different sampling strategies tailored to their specific needs, enhancing the adaptability and efficiency of the sampling process.
## Input types
### Required
- **`model`**
    - The 'model' input type specifies the model to be used for sampling, playing a crucial role in determining the sampling behavior and output.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`add_noise`**
    - The 'add_noise' input type allows users to specify whether noise should be added to the sampling process, influencing the diversity and characteristics of the generated samples.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - The 'noise_seed' input type provides a seed for the noise generation, ensuring reproducibility and consistency in the sampling process when adding noise.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The 'cfg' input type sets the configuration for the sampling process, allowing for fine-tuning of the sampling parameters and behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`positive`**
    - The 'positive' input type represents positive conditioning information, guiding the sampling process towards generating samples that align with specified positive attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`negative`**
    - The 'negative' input type represents negative conditioning information, steering the sampling process away from generating samples that exhibit specified negative attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`sampler`**
    - The 'sampler' input type selects the specific sampling strategy to be employed, directly impacting the nature and quality of the generated samples.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `str`
- **`sigmas`**
    - The 'sigmas' input type defines the noise levels to be used in the sampling process, affecting the exploration of the sample space and the diversity of the output.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`latent_image`**
    - The 'latent_image' input type provides an initial latent image for the sampling process, serving as a starting point for sample generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
## Output types
- **`output`**
    - Comfy dtype: `LATENT`
    - The 'output' represents the primary result of the sampling process, containing the generated samples.
    - Python dtype: `torch.Tensor`
- **`denoised_output`**
    - Comfy dtype: `LATENT`
    - The 'denoised_output' represents the samples after a denoising process has been applied, potentially enhancing the clarity and quality of the generated samples.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)
    - Preview Chooser
    - [LatentUpscaleBy](../../Comfy/Nodes/LatentUpscaleBy.md)



## Source code
```python
class SamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        if not add_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

```
