# StableZero123_Conditioning_Batched
## Documentation
- Class name: `StableZero123_Conditioning_Batched`
- Category: `conditioning/3d_models`
- Output node: `False`

This node is designed to process conditioning information in a batched manner specifically tailored for the StableZero123 model. It focuses on efficiently handling multiple sets of conditioning data simultaneously, optimizing the workflow for scenarios where batch processing is crucial.
## Input types
### Required
- **`clip_vision`**
    - The CLIP vision embeddings that provide visual context for the conditioning process.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.Tensor`
- **`init_image`**
    - The initial image to be conditioned upon, serving as a starting point for the generation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The variational autoencoder used for encoding and decoding images in the conditioning process.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`width`**
    - The width of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - The number of conditioning sets to be processed in a single batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`elevation`**
    - The elevation angle for 3D model conditioning, affecting the perspective of the generated image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`azimuth`**
    - The azimuth angle for 3D model conditioning, affecting the orientation of the generated image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`elevation_batch_increment`**
    - The incremental change in elevation angle across the batch, allowing for varied perspectives.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`azimuth_batch_increment`**
    - The incremental change in azimuth angle across the batch, allowing for varied orientations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning output, tailored for promoting certain features or aspects in the generated content.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning output, tailored for demoting certain features or aspects in the generated content.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation derived from the conditioning process, ready for further processing or generation steps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StableZero123_Conditioning_Batched:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "elevation_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth_batch_increment": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/3d_models"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth, elevation_batch_increment, azimuth_batch_increment):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        cam_embeds = []
        for i in range(batch_size):
            cam_embeds.append(camera_embeddings(elevation, azimuth))
            elevation += elevation_batch_increment
            azimuth += azimuth_batch_increment

        cam_embeds = torch.cat(cam_embeds, dim=0)
        cond = torch.cat([comfy.utils.repeat_to_batch_size(pooled, batch_size), cam_embeds], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent, "batch_index": [0] * batch_size})

```
