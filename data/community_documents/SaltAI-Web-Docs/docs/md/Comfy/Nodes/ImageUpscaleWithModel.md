# Upscale Image (using Model)
## Documentation
- Class name: `ImageUpscaleWithModel`
- Category: `image/upscaling`
- Output node: `False`

This node is designed for upscaling images using a specified upscale model. It handles the upscaling process by adjusting the image to the appropriate device, managing memory efficiently, and applying the upscale model in a tiled manner to accommodate for potential out-of-memory errors.
## Input types
### Required
- **`upscale_model`**
    - The upscale model to be used for upscaling the image. It is crucial for defining the upscaling algorithm and its parameters.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `torch.nn.Module`
- **`image`**
    - The image to be upscaled. This input is essential for determining the source content that will undergo the upscaling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The upscaled image, processed by the upscale model. This output is the result of the upscaling operation, showcasing the enhanced resolution or quality.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageScaleBy](../../Comfy/Nodes/ImageScaleBy.md)
    - [ImageScale](../../Comfy/Nodes/ImageScale.md)
    - [SaveImage](../../Comfy/Nodes/SaveImage.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - Reroute
    - [ImageScaleToTotalPixels](../../Comfy/Nodes/ImageScaleToTotalPixels.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [CR Image Output](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Image Output.md)
    - [ImageCASharpening+](../../ComfyUI_essentials/Nodes/ImageCASharpening+.md)



## Source code
```python
class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        free_memory = model_management.get_free_memory(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

```
