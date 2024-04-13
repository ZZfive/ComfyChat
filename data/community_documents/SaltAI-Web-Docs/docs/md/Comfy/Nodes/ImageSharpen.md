# ImageSharpen
## Documentation
- Class name: `ImageSharpen`
- Category: `image/postprocessing`
- Output node: `False`

The ImageSharpen node enhances the clarity of an image by accentuating its edges and details. It applies a sharpening filter to the image, which can be adjusted in intensity and radius, thereby making the image appear more defined and crisp.
## Input types
### Required
- **`image`**
    - The input image to be sharpened. This parameter is crucial as it determines the base image on which the sharpening effect will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`sharpen_radius`**
    - Defines the radius of the sharpening effect. A larger radius means that more pixels around the edge will be affected, leading to a more pronounced sharpening effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma`**
    - Controls the spread of the sharpening effect. A higher sigma value results in a smoother transition at the edges, while a lower sigma makes the sharpening more localized.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`alpha`**
    - Adjusts the intensity of the sharpening effect. Higher alpha values result in a stronger sharpening effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The sharpened image, with enhanced edges and details, ready for further processing or display.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)
    - [IPAdapterApply](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApply.md)
    - [ImageUpscaleWithModel](../../Comfy/Nodes/ImageUpscaleWithModel.md)
    - UltimateSDUpscale
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)



## Source code
```python
class Sharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "image/postprocessing"

    def sharpen(self, image: torch.Tensor, sharpen_radius: int, sigma:float, alpha: float):
        if sharpen_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = sharpen_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device) * -(alpha*10)
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        tensor_image = F.pad(tensor_image, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return (result,)

```
