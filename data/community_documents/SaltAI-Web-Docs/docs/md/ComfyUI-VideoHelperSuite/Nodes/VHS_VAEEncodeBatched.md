# VAE Encode Batched 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_VAEEncodeBatched`
- Category: `Video Helper Suite 🎥🅥🅗🅢/batched nodes`
- Output node: `False`

This node is designed for batch processing of images through a Variational Autoencoder (VAE) to encode them into a latent space representation. It efficiently handles large sets of images by dividing them into smaller batches, encoding each batch separately, and then aggregating the results. This approach optimizes resource utilization and accelerates the encoding process.
## Input types
### Required
- **`pixels`**
    - The 'pixels' parameter represents the images to be encoded into the latent space. It plays a crucial role in the node's operation by serving as the primary input for the encoding process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The 'vae' parameter specifies the Variational Autoencoder model used for encoding the images. It is essential for determining how the images are transformed into their latent space representations.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`per_batch`**
    - The 'per_batch' parameter controls the number of images processed in each batch. Adjusting this value can optimize the encoding process by balancing between processing speed and resource consumption.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a dictionary containing the encoded images in their latent space representation, facilitating further processing or analysis.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VAEEncodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", ), "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    def encode(self, vae, pixels, per_batch):
        t = []
        for start_idx in range(0, pixels.shape[0], per_batch):
            sub_pixels = VAEEncode.vae_encode_crop_pixels(pixels[start_idx:start_idx+per_batch])
            t.append(vae.encode(sub_pixels[:,:,:,:3]))
        return ({"samples": torch.cat(t, dim=0)}, )

```
