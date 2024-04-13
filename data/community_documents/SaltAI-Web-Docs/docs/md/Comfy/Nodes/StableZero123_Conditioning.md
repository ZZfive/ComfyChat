# StableZero123_Conditioning
## Documentation
- Class name: `StableZero123_Conditioning`
- Category: `conditioning/3d_models`
- Output node: `False`

This node is designed to process and condition data for use in StableZero123 models, focusing on preparing the input in a specific format that is compatible and optimized for these models.
## Input types
### Required
- **`clip_vision`**
    - Processes visual data to align with the model's requirements, enhancing the model's understanding of visual context.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.Tensor`
- **`init_image`**
    - Serves as the initial image input for the model, setting the baseline for further image-based operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Integrates variational autoencoder outputs, facilitating the model's ability to generate or modify images.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`width`**
    - Specifies the width of the output image, allowing for dynamic resizing according to model needs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output image, enabling customization of the output dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Controls the number of images processed in a single batch, optimizing computational efficiency.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`elevation`**
    - Adjusts the elevation angle for 3D model rendering, enhancing the model's spatial understanding.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`azimuth`**
    - Modifies the azimuth angle for 3D model visualization, improving the model's perception of orientation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Generates positive conditioning vectors, aiding in the model's positive feature reinforcement.
    - Python dtype: `List[torch.Tensor]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Produces negative conditioning vectors, assisting in the model's avoidance of certain features.
    - Python dtype: `List[torch.Tensor]`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Creates latent representations, facilitating deeper model insights into the data.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)



## Source code
```python
class StableZero123_Conditioning:
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
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "conditioning/3d_models"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.to(pooled.device).repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

```
