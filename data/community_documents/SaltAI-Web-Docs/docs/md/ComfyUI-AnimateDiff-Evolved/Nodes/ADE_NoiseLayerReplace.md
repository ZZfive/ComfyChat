# Noise Layer [Replace] 🎭🅐🅓
## Documentation
- Class name: `ADE_NoiseLayerReplace`
- Category: `Animate Diff 🎭🅐🅓/noise layers`
- Output node: `False`

This node specializes in replacing specific noise layers within a given noise structure, utilizing parameters such as noise type, batch offset, and seed generation overrides. It aims to modify the noise characteristics by selectively replacing parts of the noise based on a mask, thereby enabling fine-tuned control over the noise's impact on the generation process.
## Input types
### Required
- **`batch_offset`**
    - Determines the offset for batch processing, influencing how noise is applied across different batches.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_type`**
    - Specifies the type of noise to be used in the replacement process, affecting the characteristics of the generated noise.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_gen_override`**
    - Overrides the default seed generation mechanism, allowing for customized noise generation based on the provided seed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_offset`**
    - Adjusts the seed value used in noise generation, enabling variations in the noise pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`prev_noise_layers`**
    - The previous noise layers that will be modified by the replacement process.
    - Comfy dtype: `NOISE_LAYERS`
    - Python dtype: `NoiseLayerGroup`
- **`mask_optional`**
    - An optional tensor mask that can be used for selective noise replacement, providing additional flexibility in the modification process. Despite its name suggesting optionality, it is a required parameter for the operation of this node.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`seed_override`**
    - Directly specifies a seed value to be used, bypassing the normal seed generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`noise_layers`**
    - Comfy dtype: `NOISE_LAYERS`
    - The updated noise layers after the replacement process, incorporating the modifications made.
    - Python dtype: `NoiseLayerGroup`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class NoiseLayerReplaceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen_override": (SeedNoiseGeneration.LIST_WITH_OVERRIDE,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
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
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,):
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerReplace(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                                  seed_override=seed_override, mask=mask_optional)
        prev_noise_layers.add_to_start(layer)
        return (prev_noise_layers,)

```
