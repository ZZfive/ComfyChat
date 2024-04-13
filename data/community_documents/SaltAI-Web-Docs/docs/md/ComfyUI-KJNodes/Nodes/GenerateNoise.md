# GenerateNoise
## Documentation
- Class name: `GenerateNoise`
- Category: `KJNodes/noise`
- Output node: `False`

The GenerateNoise node is designed to create a batch of noise samples with specified dimensions and properties. It allows for the generation of noise with controlled variability through parameters like seed and multiplier, and offers options for normalization and constant batch noise, making it versatile for various generative tasks.
## Input types
### Required
- **`width`**
    - Determines the width of each noise sample, influencing the spatial dimensions of the generated noise.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of each noise sample, impacting the spatial dimensions of the generated noise.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Specifies the number of noise samples to generate in a batch, affecting the overall volume of data produced.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - Controls the randomness of noise generation, ensuring reproducibility of results when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Applies a scaling factor to the noise, adjusting its amplitude.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`constant_batch_noise`**
    - When enabled, uses the same noise sample across the entire batch, ensuring uniformity in noise characteristics.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`normalize`**
    - Normalizes the noise samples, typically to have a standard deviation of 1, enhancing consistency in their distribution.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`model`**
    - An optional model parameter that, if provided, is used to further adjust the noise based on the model's characteristics.
    - Comfy dtype: `MODEL`
    - Python dtype: `Optional[torch.nn.Module]`
- **`sigmas`**
    - Optional parameter for adjusting the noise intensity based on a range of sigma values, allowing for dynamic control over the noise's variability.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The generated noise samples, ready for use in various generative tasks.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GenerateNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 4096, "step": 0.01}),
            "constant_batch_noise": ("BOOLEAN", {"default": False}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            "model": ("MODEL", ),
            "sigmas": ("SIGMAS", ),
            }
            }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generatenoise"

    CATEGORY = "KJNodes/noise"
        
    def generatenoise(self, batch_size, width, height, seed, multiplier, constant_batch_noise, normalize, sigmas=None, model=None):

        generator = torch.manual_seed(seed)
        noise = torch.randn([batch_size, 4, height // 8, width // 8], dtype=torch.float32, layout=torch.strided, generator=generator, device="cpu")
        if sigmas is not None:
            sigma = sigmas[0] - sigmas[-1]
            sigma /= model.model.latent_format.scale_factor
            noise *= sigma

        noise *=multiplier

        if normalize:
            noise = noise / noise.std()
        if constant_batch_noise:
            noise = noise[0].repeat(batch_size, 1, 1, 1)
        return ({"samples":noise}, )

```
