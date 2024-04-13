# Model Patch Seamless (mtb)
## Documentation
- Class name: `Model Patch Seamless (mtb)`
- Category: `mtb/textures`
- Output node: `False`

This node leverages a stable diffusion technique to generate seamless images by modifying the padding mode of model layers to circular. It's an experimental approach aimed at enhancing image synthesis for textures and patterns, ensuring continuity across edges.
## Input types
### Required
- **`model`**
    - The model to be patched for seamless image generation. It's crucial as it undergoes modifications to its convolutional layers to achieve the desired effect.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`tiling`**
    - A boolean flag indicating whether to enable or disable the circular padding mode. It affects the continuity of the generated images, playing a key role in achieving seamless textures.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`Original Model (passthrough)`**
    - Comfy dtype: `MODEL`
    - The original input model, passed through without modifications.
    - Python dtype: `torch.nn.Module`
- **`Patched Model`**
    - Comfy dtype: `MODEL`
    - The modified version of the input model, with its padding mode set to circular for seamless image generation.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ModelPatchSeamless:
    """Uses the stable diffusion 'hack' to infer seamless images by setting the model layers padding mode to circular (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (
                    "BOOLEAN",
                    {"default": True},
                ),  # kept for testing not sure why it should be false
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = (
        "Original Model (passthrough)",
        "Patched Model",
    )
    FUNCTION = "hack"

    CATEGORY = "mtb/textures"

    def apply_circular(self, model, enable):
        for layer in [
            layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
        ]:
            layer.padding_mode = "circular" if enable else "zeros"
        return model

    def hack(
        self,
        model,
        tiling,
    ):
        hacked_model = copy.deepcopy(model)
        self.apply_circular(hacked_model.model, tiling)
        return (model, hacked_model)

```
