# RGB SparseCtrl 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlRGBPreprocessor`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess`
- Output node: `False`

This node is designed for preprocessing images in the context of Sparse Control Networks, specifically for RGB Sparse Control. It adjusts the input image to match the latent dimensions required by the network, encodes the image using a VAE (Variational Autoencoder), and wraps the encoded latent representation in a special wrapper that ensures it is directly compatible with Apply ControlNet nodes. This preprocessing step is crucial for integrating image data into the Sparse Control workflow, particularly when dealing with RGB images.
## Input types
### Required
- **`image`**
    - The input image to be preprocessed. This image is resized and encoded to match the latent dimensions required by the network.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The Variational Autoencoder (VAE) used for encoding the input image into a latent representation. It is crucial for transforming the image into a format suitable for the Sparse Control workflow.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`latent_size`**
    - Specifies the target size for the latent representation. This size is used to resize the input image appropriately before encoding.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
## Output types
- **`proc_IMAGE`**
    - Comfy dtype: `IMAGE`
    - The processed image, wrapped in a PreprocSparseRGBWrapper, ensuring it is ready for direct use with Apply ControlNet nodes.
    - Python dtype: `PreprocSparseRGBWrapper`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RgbSparseCtrlPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", ),
                "latent_size": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("proc_IMAGE",)
    FUNCTION = "preprocess_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess"

    def preprocess_images(self, vae, image: Tensor, latent_size: Tensor):
        # first, resize image to match latents
        image = image.movedim(-1,1)
        image = comfy.utils.common_upscale(image, latent_size["samples"].shape[3] * 8, latent_size["samples"].shape[2] * 8, 'nearest-exact', "center")
        image = image.movedim(1,-1)
        # then, vae encode
        image = VAEEncode.vae_encode_crop_pixels(image)
        encoded = vae.encode(image[:,:,:,:3])
        return (PreprocSparseRGBWrapper(condhint=encoded),)

```
