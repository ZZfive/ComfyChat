# BasicScheduler
## Documentation
- Class name: `BasicScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The BasicScheduler node is designed to compute a sequence of sigma values for diffusion models based on the provided scheduler, model, and denoising parameters. It dynamically adjusts the total number of steps based on the denoise factor to fine-tune the diffusion process.
## Input types
### Required
- **`model`**
    - The model parameter specifies the diffusion model for which the sigma values are to be calculated. It plays a crucial role in determining the appropriate sigma values for the diffusion process.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_management.Model`
- **`scheduler`**
    - The scheduler parameter determines the scheduling algorithm to be used for calculating the sigma values. It directly influences the progression and characteristics of the diffusion process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`steps`**
    - The steps parameter indicates the total number of steps in the diffusion process. It affects the granularity and duration of the process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - The denoise parameter allows for adjusting the effective number of steps by scaling the total steps, enabling finer control over the diffusion process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The sigmas output represents the computed sequence of sigma values for the diffusion process, essential for controlling the noise level at each step.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [SplitSigmas](../../Comfy/Nodes/SplitSigmas.md)
    - Reroute



## Source code
```python
class BasicScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps/denoise)

        comfy.model_management.load_models_gpu([model])
        sigmas = comfy.samplers.calculate_sigmas_scheduler(model.model, scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return (sigmas, )

```
