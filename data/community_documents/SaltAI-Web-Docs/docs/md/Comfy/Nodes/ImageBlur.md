# ImageBlur
## Documentation
- Class name: `ImageBlur`
- Category: `image/postprocessing`
- Output node: `False`

The ImageBlur node applies a Gaussian blur to an image, allowing for the softening of edges and reduction of detail and noise. It provides control over the intensity and spread of the blur through parameters.
## Input types
### Required
- **`image`**
    - The input image to be blurred. This is the primary target for the blur effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`blur_radius`**
    - Determines the radius of the blur effect. A larger radius results in a more pronounced blur.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma`**
    - Controls the spread of the blur. A higher sigma value means the blur will affect a wider area around each pixel.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is the blurred version of the input image, with the degree of blur determined by the input parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageUpscaleWithModel](../../Comfy/Nodes/ImageUpscaleWithModel.md)
    - [Cut By Mask](../../masquerade-nodes-comfyui/Nodes/Cut By Mask.md)



## Source code
```python
class Blur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "image/postprocessing"

    def blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        padded_image = F.pad(image, (blur_radius,blur_radius,blur_radius,blur_radius), 'reflect')
        blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:,:,blur_radius:-blur_radius, blur_radius:-blur_radius]
        blurred = blurred.permute(0, 2, 3, 1)

        return (blurred,)

```
