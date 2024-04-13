# StableCascade_StageC_VAEEncode
## Documentation
- Class name: `StableCascade_StageC_VAEEncode`
- Category: `_for_testing/stable_cascade`
- Output node: `False`

This node is designed to encode an image into a latent representation using a specified VAE model, with an option to adjust the compression level. It primarily serves the purpose of transforming visual data into a more compact, latent form that can be further processed or manipulated within a stable cascade framework.
## Input types
### Required
- **`image`**
    - The image to be encoded into a latent representation. It is crucial for the encoding process as it represents the visual data to be transformed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The VAE model used for encoding the image. It defines the specific encoding mechanism and parameters.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE class instance`
- **`compression`**
    - Specifies the level of compression applied during the encoding process. It directly influences the dimensions of the output latent representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`stage_c`**
    - Comfy dtype: `LATENT`
    - The encoded latent representation of the input image, specifically for stage C of the stable cascade process.
    - Python dtype: `Dict[str, torch.Tensor]`
- **`stage_b`**
    - Comfy dtype: `LATENT`
    - A placeholder latent representation for stage B, initialized to zeros.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class StableCascade_StageC_VAEEncode:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "compression": ("INT", {"default": 42, "min": 4, "max": 128, "step": 1}),
        }}
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "_for_testing/stable_cascade"

    def generate(self, image, vae, compression):
        width = image.shape[-2]
        height = image.shape[-3]
        out_width = (width // compression) * vae.downscale_ratio
        out_height = (height // compression) * vae.downscale_ratio

        s = comfy.utils.common_upscale(image.movedim(-1,1), out_width, out_height, "bicubic", "center").movedim(1,-1)

        c_latent = vae.encode(s[:,:,:,:3])
        b_latent = torch.zeros([c_latent.shape[0], 4, height // 4, width // 4])
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

```
