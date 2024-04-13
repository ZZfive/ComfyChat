# Sharpen (mtb)
## Documentation
- Class name: `Sharpen (mtb)`
- Category: `mtb/image processing`
- Output node: `False`

The Sharpen node enhances the details of an image by applying a sharpening filter. This process involves using a Gaussian kernel to accentuate edges and fine details, making the image appear more crisp and defined.
## Input types
### Required
- **`image`**
    - The input image to be sharpened. This is the primary data upon which the sharpening process is applied, affecting the visual clarity and detail of the output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`sharpen_radius`**
    - Defines the radius of the sharpening effect. A larger radius results in a more pronounced sharpening effect, impacting the overall sharpness of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_x`**
    - The standard deviation in the x-direction for the Gaussian kernel. This parameter influences the spread of the sharpening effect horizontally.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_y`**
    - The standard deviation in the y-direction for the Gaussian kernel. This parameter influences the spread of the sharpening effect vertically.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`alpha`**
    - Controls the intensity of the sharpening effect. A higher alpha value results in a stronger sharpening effect, enhancing the contrast between edges and the surrounding areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The sharpened image. This output showcases the enhanced details and edges, resulting from the application of the sharpening filter.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Sharpen_:
    """Sharpens an image using a Gaussian kernel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": (
                    "INT",
                    {"default": 1, "min": 1, "max": 31, "step": 1},
                ),
                "sigma_x": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "sigma_y": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_sharp"
    CATEGORY = "mtb/image processing"

    def do_sharp(
        self,
        image: torch.Tensor,
        sharpen_radius: int,
        sigma_x: float,
        sigma_y: float,
        alpha: float,
    ):
        if sharpen_radius == 0:
            return (image,)

        channels = image.shape[3]

        kernel_size = 2 * sharpen_radius + 1
        kernel = gaussian_kernel(kernel_size, sigma_x, sigma_y) * -(alpha * 10)

        # Modify center of kernel to make it a sharpening kernel
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0

        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)
        tensor_image = image.permute(0, 3, 1, 2)

        tensor_image = F.pad(
            tensor_image,
            (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
            "reflect",
        )
        sharpened = F.conv2d(
            tensor_image, kernel, padding=center, groups=channels
        )

        # Remove padding
        sharpened = sharpened[
            :,
            :,
            sharpen_radius:-sharpen_radius,
            sharpen_radius:-sharpen_radius,
        ]

        sharpened = sharpened.permute(0, 2, 3, 1)
        result = torch.clamp(sharpened, 0, 1)

        return (result,)

```
