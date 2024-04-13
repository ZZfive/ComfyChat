# PolyexponentialScheduler
## Documentation
- Class name: `PolyexponentialScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The PolyexponentialScheduler node is designed to generate a sequence of noise levels (sigmas) based on a polyexponential noise schedule. This schedule is a polynomial function in the logarithm of sigma, allowing for a flexible and customizable progression of noise levels throughout the diffusion process.
## Input types
### Required
- **`steps`**
    - Specifies the number of steps in the diffusion process, affecting the granularity of the generated noise levels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_max`**
    - The maximum noise level, setting the upper bound of the noise schedule.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_min`**
    - The minimum noise level, setting the lower bound of the noise schedule.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rho`**
    - A parameter that controls the shape of the polyexponential noise schedule, influencing how noise levels progress between the minimum and maximum values.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The output is a sequence of noise levels (sigmas) tailored to the specified polyexponential noise schedule.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PolyexponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

```
