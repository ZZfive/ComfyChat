# KarrasScheduler
## Documentation
- Class name: `KarrasScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The KarrasScheduler node is designed to generate a sequence of noise levels (sigmas) based on the Karras et al. (2022) noise schedule. This scheduler is useful for controlling the diffusion process in generative models, allowing for fine-tuned adjustments to the noise levels applied at each step of the generation process.
## Input types
### Required
- **`steps`**
    - Specifies the number of steps in the noise schedule, affecting the granularity of the generated sigmas sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_max`**
    - The maximum sigma value in the noise schedule, setting the upper bound of noise levels.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_min`**
    - The minimum sigma value in the noise schedule, setting the lower bound of noise levels.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rho`**
    - A parameter that controls the shape of the noise schedule curve, influencing how noise levels progress from sigma_min to sigma_max.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The generated sequence of noise levels (sigmas) following the Karras et al. (2022) noise schedule.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KarrasScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

```
