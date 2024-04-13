# FlipSigmas
## Documentation
- Class name: `FlipSigmas`
- Category: `sampling/custom_sampling/sigmas`
- Output node: `False`

The FlipSigmas node is designed to manipulate the sequence of sigma values used in diffusion models by reversing their order and ensuring the first value is non-zero if originally zero. This operation is crucial for adapting the noise levels in reverse order, facilitating the generation process in models that operate by gradually reducing noise from data.
## Input types
### Required
- **`sigmas`**
    - The 'sigmas' parameter represents the sequence of sigma values to be flipped. This sequence is crucial for controlling the noise levels applied during the diffusion process, and flipping it is essential for the reverse generation process.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The output is the modified sequence of sigma values, flipped and adjusted to ensure the first value is non-zero if originally zero, ready for use in subsequent diffusion model operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FlipSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas):
        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        return (sigmas,)

```
