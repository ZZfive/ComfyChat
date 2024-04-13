# FlipSigmasAdjusted
## Documentation
- Class name: `FlipSigmasAdjusted`
- Category: `KJNodes/noise`
- Output node: `False`

The FlipSigmasAdjusted node is designed to adjust the sequence of sigma values used in diffusion models. It flips the order of the sigma values, ensuring that the sequence starts with the smallest value. If the first sigma value is zero, it is replaced with a small positive value to avoid division by zero errors. Additionally, it adjusts subsequent sigma values based on the previous ones to maintain the integrity of the diffusion process.
## Input types
### Required
- **`sigmas`**
    - The 'sigmas' parameter represents the sequence of sigma values used in the diffusion process. Adjusting these values is crucial for controlling the noise level at each step of the diffusion, which in turn affects the quality and characteristics of the generated images.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The output is an adjusted sequence of sigma values, modified to ensure a proper diffusion process by flipping the order and making necessary adjustments to the values.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FlipSigmasAdjusted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "KJNodes/noise"

    FUNCTION = "get_sigmas_adjusted"

    def get_sigmas_adjusted(self, sigmas):
        
        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        adjusted_sigmas = sigmas.clone()
        #offset sigma
        for i in range(1, len(sigmas)):
            adjusted_sigmas[i] = sigmas[i - 1]

        if adjusted_sigmas[0] == 0:
            adjusted_sigmas[0] = 0.0001  
        
        return (adjusted_sigmas,)

```
