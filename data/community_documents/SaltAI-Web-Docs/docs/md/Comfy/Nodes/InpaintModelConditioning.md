# InpaintModelConditioning
## Documentation
- Class name: `InpaintModelConditioning`
- Category: `conditioning/inpaint`
- Output node: `False`

The InpaintModelConditioning node is designed to facilitate the conditioning process for inpainting models, enabling the integration and manipulation of various conditioning inputs to tailor the inpainting output. It encompasses a broad range of functionalities, from loading specific model checkpoints and applying style or control net models, to encoding and combining conditioning elements, thereby serving as a comprehensive tool for customizing inpainting tasks.
## Input types
### Required
- **`positive`**
    - Represents the positive conditioning information or parameters that are to be applied to the inpainting model. This input is crucial for defining the context or constraints under which the inpainting operation should be performed, affecting the final output significantly.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`negative`**
    - Represents the negative conditioning information or parameters that are to be applied to the inpainting model. This input is essential for specifying the conditions or contexts to avoid during the inpainting process, thereby influencing the final output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`vae`**
    - Specifies the VAE model to be used in the conditioning process. This input is crucial for determining the specific architecture and parameters of the VAE model that will be utilized.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`pixels`**
    - Represents the pixel data of the image to be inpainted. This input is essential for providing the visual context necessary for the inpainting task.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Specifies the mask to be applied to the image, indicating the areas to be inpainted. This input is crucial for defining the specific regions within the image that require inpainting.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The modified positive conditioning information after processing, ready to be applied to the inpainting model. This output is essential for guiding the inpainting process according to the specified positive conditions.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified negative conditioning information after processing, ready to be applied to the inpainting model. This output is essential for guiding the inpainting process according to the specified negative conditions.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation derived from the conditioning process. This output is crucial for understanding the underlying features and characteristics of the image being inpainted.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InpaintModelConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "pixels": ("IMAGE", ),
                             "mask": ("MASK", ),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, positive, negative, pixels, vae, mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                d["concat_mask"] = mask
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], out_latent)

```
