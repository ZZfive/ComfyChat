# ExponentialScheduler
## Documentation
- Class name: `ExponentialScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The ExponentialScheduler node is designed to generate a sequence of sigma values following an exponential schedule for diffusion sampling processes. It provides a customizable approach to control the noise levels applied at each step of the diffusion process, allowing for fine-tuning of the sampling behavior.
## Input types
### Required
- **`steps`**
    - Specifies the number of steps in the diffusion process. It influences the length of the generated sigma sequence and thus the granularity of the noise application.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_max`**
    - Defines the maximum sigma value, setting the upper limit of noise intensity in the diffusion process. It plays a crucial role in determining the range of noise levels applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_min`**
    - Sets the minimum sigma value, establishing the lower boundary of noise intensity. This parameter helps in fine-tuning the starting point of the noise application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - A sequence of sigma values generated according to the exponential schedule. These values are used to control the noise levels at each step of the diffusion process.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ExponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min):
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        return (sigmas, )

```
