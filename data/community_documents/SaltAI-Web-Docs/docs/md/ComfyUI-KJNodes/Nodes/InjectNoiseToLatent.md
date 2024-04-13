# InjectNoiseToLatent
## Documentation
- Class name: `InjectNoiseToLatent`
- Category: `KJNodes/noise`
- Output node: `False`

This node is designed to inject noise into latent representations, offering options for normalization, averaging, and selective application through a mask. It enhances the diversity and quality of generated samples by blending original latents with noise in a controlled manner.
## Input types
### Required
- **`latents`**
    - The original latent representations to which noise will be added. This input is crucial for defining the base upon which noise injection and subsequent modifications are applied.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`strength`**
    - A scalar multiplier for the noise, adjusting its intensity before addition to the latents. This parameter allows for fine-tuning the impact of noise on the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - The noise to be injected into the latents. This input provides the variability necessary for the node's operation, directly influencing the diversity of the output.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`normalize`**
    - A boolean flag that determines whether the noised latents should be normalized. Normalization can help maintain the statistical properties of the latents post-noise injection.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`average`**
    - A boolean flag indicating whether the noise should be averaged with the original latents. Averaging can create a more subtle effect of noise on the latents.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`mask`**
    - An optional mask that specifies where noise should be applied, allowing for selective noise injection. This enables targeted modifications and can be used for tasks like inpainting. It is not required for the node's operation but when provided, it allows for more precise control over where the noise affects the latents.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The modified latent representations with noise injected, potentially normalized, and selectively applied according to the provided mask.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InjectNoiseToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents":("LATENT",),  
            "strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 200.0, "step": 0.0001}),
            "noise":  ("LATENT",),
            "normalize": ("BOOLEAN", {"default": False}),
            "average": ("BOOLEAN", {"default": False}),
            },
            "optional":{
                "mask": ("MASK", ),
            }
            }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "injectnoise"

    CATEGORY = "KJNodes/noise"
        
    def injectnoise(self, latents, strength, noise, normalize, average, mask=None):
        samples = latents.copy()
        if latents["samples"].shape != noise["samples"].shape:
            raise ValueError("InjectNoiseToLatent: Latent and noise must have the same shape")
        if average:
            noised = (samples["samples"].clone() + noise["samples"].clone()) / 2
        else:
            noised = samples["samples"].clone() + noise["samples"].clone() * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noised.shape[2], noised.shape[3]), mode="bilinear")
            mask = mask.expand((-1,noised.shape[1],-1,-1))
            if mask.shape[0] < noised.shape[0]:
                mask = mask.repeat((noised.shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:noised.shape[0]]
            noised = mask * noised + (1-mask) * latents["samples"]
        samples["samples"] = noised
        return (samples,)

```
