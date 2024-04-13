# VPScheduler
## Documentation
- Class name: `VPScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The VPScheduler node is designed to generate a sequence of noise levels (sigmas) based on the Variance Preserving (VP) scheduling method. This sequence is crucial for guiding the denoising process in diffusion models, allowing for controlled generation of images or other data types.
## Input types
### Required
- **`steps`**
    - Specifies the number of steps in the diffusion process, affecting the granularity of the generated noise levels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`beta_d`**
    - Determines the overall noise level distribution, influencing the variance of the generated noise levels.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`beta_min`**
    - Sets the minimum boundary for the noise level, ensuring the noise does not fall below a certain threshold.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`eps_s`**
    - Adjusts the starting epsilon value, fine-tuning the initial noise level in the diffusion process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - A sequence of noise levels (sigmas) generated based on the VP scheduling method, used to guide the denoising process in diffusion models.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class VPScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}), #TODO: fix default values
                     "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1000.0, "step":0.01, "round": False}),
                     "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step":0.0001, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, beta_d, beta_min, eps_s):
        sigmas = k_diffusion_sampling.get_sigmas_vp(n=steps, beta_d=beta_d, beta_min=beta_min, eps_s=eps_s)
        return (sigmas, )

```
