# Noise Layer [Add] 🎭🅐🅓
## Documentation
- Class name: `ADE_NoiseLayerAdd`
- Category: `Animate Diff 🎭🅐🅓/noise layers`
- Output node: `False`

The ADE_NoiseLayerAdd node is designed to introduce additive noise layers into a given input, enhancing the variability and richness of the output through controlled noise manipulation. It allows for the dynamic adjustment of noise characteristics, enabling a more nuanced and customizable generation process.
## Input types
### Required
- **`batch_offset`**
    - Determines the offset for batch processing, allowing for sequential noise layer application across different batches.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_type`**
    - Specifies the type of noise to be added, influencing the texture and characteristics of the resulting noise layer.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_gen_override`**
    - Overrides the default seed generator, enabling the use of a custom seed for noise generation which can affect the randomness and pattern of the noise.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_offset`**
    - Applies an additional offset to the seed value, further customizing the randomness and distribution of the noise.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_weight`**
    - Controls the intensity of the added noise, allowing for fine-tuning of the noise layer's impact on the overall output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_noise_layers`**
    - Optionally includes previous noise layers for cumulative noise layer application, enhancing the depth and complexity of the noise effect.
    - Comfy dtype: `NOISE_LAYERS`
    - Python dtype: `NoiseLayerGroup`
- **`mask_optional`**
    - Optionally applies a mask to selectively add noise, enabling targeted noise application for specific areas or features.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`seed_override`**
    - Allows for the explicit specification of a seed value, overriding the automatic seed generation for precise control over the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`noise_layers`**
    - Comfy dtype: `NOISE_LAYERS`
    - Returns the collection of noise layers, including the newly added layer, facilitating the construction of complex noise layer sequences.
    - Python dtype: `NoiseLayerGroup`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class NoiseLayerAddNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
                "noise_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_noise_layers": ("NOISE_LAYERS",),
                "mask_optional": ("MASK",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("NOISE_LAYERS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/noise layers"
    FUNCTION = "create_layers"

    def create_layers(self, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,):
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerAdd(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)

```
