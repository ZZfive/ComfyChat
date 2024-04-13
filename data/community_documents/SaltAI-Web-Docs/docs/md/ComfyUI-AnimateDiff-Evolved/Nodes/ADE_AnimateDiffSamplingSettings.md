# Sample Settings 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffSamplingSettings`
- Category: `Animate Diff 🎭🅐🅓`
- Output node: `False`

The ADE_AnimateDiffSamplingSettings node is designed to configure the sampling settings for the AnimateDiff process, allowing users to specify parameters such as batch offset, noise type, and seed generation. This node plays a crucial role in tailoring the sampling behavior to achieve desired animation effects in generated images.
## Input types
### Required
- **`batch_offset`**
    - Specifies the offset for the batch in the sampling process, affecting the starting point of sampling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_type`**
    - Determines the type of noise to be applied during the sampling process, influencing the visual characteristics of the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_gen`**
    - Defines the method for generating seeds used in the noise generation, impacting the randomness and variation in the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_offset`**
    - Sets the offset for the seed generation, modifying the starting point for random noise generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`noise_layers`**
    - Optional parameter that allows specifying custom noise layers for more granular control over the noise applied.
    - Comfy dtype: `NOISE_LAYERS`
    - Python dtype: `NoiseLayerGroup`
- **`iteration_opts`**
    - Optional parameter for setting iteration options, providing additional control over the sampling iterations.
    - Comfy dtype: `ITERATION_OPTS`
    - Python dtype: `IterationOptions`
- **`seed_override`**
    - Optional parameter to override the seed used in noise generation, enabling deterministic animations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`adapt_denoise_steps`**
    - Optional boolean parameter to adapt the number of denoising steps based on certain conditions, potentially improving animation quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`settings`**
    - Comfy dtype: `SAMPLE_SETTINGS`
    - The configured sampling settings ready to be used in the AnimateDiff process.
    - Python dtype: `SampleSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SampleSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen": (SeedNoiseGeneration.LIST,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            },
            "optional": {
                "noise_layers": ("NOISE_LAYERS",),
                "iteration_opts": ("ITERATION_OPTS",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "adapt_denoise_steps": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("SAMPLE_SETTINGS",)
    RETURN_NAMES = ("settings",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "create_settings"

    def create_settings(self, batch_offset: int, noise_type: str, seed_gen: str, seed_offset: int, noise_layers: NoiseLayerGroup=None,
                        iteration_opts: IterationOptions=None, seed_override: int=None, adapt_denoise_steps=False):
        sampling_settings = SampleSettings(batch_offset=batch_offset, noise_type=noise_type, seed_gen=seed_gen, seed_offset=seed_offset, noise_layers=noise_layers,
                                           iteration_opts=iteration_opts, seed_override=seed_override, adapt_denoise_steps=adapt_denoise_steps)
        return (sampling_settings,)

```
