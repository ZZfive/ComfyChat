# Vae Decode (mtb)
## Documentation
- Class name: `Vae Decode (mtb)`
- Category: `mtb/decode`
- Output node: `False`

The VaeDecode_ node provides a flexible decoding mechanism for variational autoencoder (VAE) models, supporting both standard and tiled decoding approaches. It incorporates a seamless model option to adjust convolutional layers for circular padding, enhancing the generation of seamless images. This node is designed to adaptively choose between decoding methods based on the input configuration, optimizing for image quality and computational efficiency.
## Input types
### Required
- **`samples`**
    - The latent representations to be decoded into images. This input is crucial for generating the final image output from the encoded latent space.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`vae`**
    - The variational autoencoder model used for decoding the samples. It defines the architecture and parameters for the decoding process.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`seamless_model`**
    - A boolean flag indicating whether to adjust the VAE's convolutional layers for circular padding, enabling seamless image generation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_tiling_decoder`**
    - A boolean flag that determines whether to use a tiling decoder for image generation, which can enhance image quality for large images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tile_size`**
    - The size of the tiles used in the tiling decoder. This parameter influences the granularity of the tiling process and can affect the final image quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The decoded images, either as a single image or a batch of images, depending on the input samples.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VaeDecode_:
    """Wrapper for the 2 core decoders but also adding the sd seamless hack, taken from: FlyingFireCo/tiled_ksampler"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "seamless_model": ("BOOLEAN", {"default": False}),
                "use_tiling_decoder": ("BOOLEAN", {"default": True}),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 320, "max": 4096, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "mtb/decode"

    def decode(
        self, vae, samples, seamless_model, use_tiling_decoder=True, tile_size=512
    ):
        if seamless_model:
            if use_tiling_decoder:
                log.error(
                    "You cannot use seamless mode with tiling decoder together, skipping tiling."
                )
                use_tiling_decoder = False
            for layer in [
                layer
                for layer in vae.first_stage_model.modules()
                if isinstance(layer, torch.nn.Conv2d)
            ]:
                layer.padding_mode = "circular"
        if use_tiling_decoder:
            return (
                vae.decode_tiled(
                    samples["samples"],
                    tile_x=tile_size // 8,
                    tile_y=tile_size // 8,
                ),
            )
        else:
            return (vae.decode(samples["samples"]),)

```
