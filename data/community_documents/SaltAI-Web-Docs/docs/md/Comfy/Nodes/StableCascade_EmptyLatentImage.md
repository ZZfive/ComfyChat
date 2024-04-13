# StableCascade_EmptyLatentImage
## Documentation
- Class name: `StableCascade_EmptyLatentImage`
- Category: `_for_testing/stable_cascade`
- Output node: `False`

The node is designed to generate empty latent images with specified dimensions and compression settings. It serves as a foundational step in generating images by providing a blank canvas in the form of latent representations.
## Input types
### Required
- **`width`**
    - Specifies the width of the generated latent image. It determines the horizontal dimension of the output latent space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the generated latent image, affecting its vertical dimension in the latent space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`compression`**
    - Controls the compression level of the latent image, influencing the granularity of the generated latent representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Indicates the number of latent images to generate in a single batch, allowing for batch processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`stage_c`**
    - Comfy dtype: `LATENT`
    - Represents the first stage latent image with a specific compression applied.
    - Python dtype: `torch.Tensor`
- **`stage_b`**
    - Comfy dtype: `LATENT`
    - Denotes the second stage latent image, typically with a different level of detail or compression.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class StableCascade_EmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "compression": ("INT", {"default": 42, "min": 4, "max": 128, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
        }}
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "_for_testing/stable_cascade"

    def generate(self, width, height, compression, batch_size=1):
        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

```
