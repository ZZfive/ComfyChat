# Image Resize Factor (mtb)
## Documentation
- Class name: `Image Resize Factor (mtb)`
- Category: `mtb/image`
- Output node: `False`

The Image Resize Factor node is designed for resizing images based on a specified scaling factor. It supports optional supersampling for enhanced quality and allows for the application of a mask to selectively resize portions of the image. This node is versatile in handling different image dimensions and resampling methods, making it suitable for various image processing tasks.
## Input types
### Required
- **`image`**
    - The input image tensor to be resized. It is crucial as it determines the content and structure of the output image after resizing.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`factor`**
    - The scaling factor for resizing the image. It directly influences the dimensions of the output image, making it a key parameter in the resizing process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`supersample`**
    - A boolean flag indicating whether supersampling should be applied. Supersampling can enhance the quality of the resized image by reducing aliasing effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`resampling`**
    - Specifies the method used for resampling during the resize operation. The choice of resampling method can affect the quality and characteristics of the resized image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask tensor that, if provided, applies the resize operation selectively based on the mask. This allows for more controlled resizing effects.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor or None`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resized image tensor, reflecting the applied scaling factor and resampling method.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask tensor, if provided, is applied to selectively resize portions of the image.
    - Python dtype: `torch.Tensor or None`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageResizeFactor:
    """Extracted mostly from WAS Node Suite, with a few edits (most notably multiple image support) and less features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": (
                    "FLOAT",
                    {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
                "supersample": ("BOOLEAN", {"default": True}),
                "resampling": (
                    [
                        "nearest",
                        "linear",
                        "bilinear",
                        "bicubic",
                        "trilinear",
                        "area",
                        "nearest-exact",
                    ],
                    {"default": "nearest"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "mtb/image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"

    def resize(
        self,
        image: torch.Tensor,
        factor: float,
        supersample: bool,
        resampling: str,
        mask=None,
    ):
        # Check if the tensor has the correct dimension
        if len(image.shape) not in [3, 4]:  # HxWxC or BxHxWxC
            raise ValueError(
                "Expected image tensor of shape (H, W, C) or (B, H, W, C)"
            )

        # Transpose to CxHxW or BxCxHxW for PyTorch
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)  # CxHxW
        else:
            image = image.permute(0, 3, 1, 2)  # BxCxHxW

        # Compute new dimensions
        B, C, H, W = image.shape
        new_H, new_W = int(H * factor), int(W * factor)

        align_corner_filters = ("linear", "bilinear", "bicubic", "trilinear")
        # Resize the image
        resized_image = F.interpolate(
            image,
            size=(new_H, new_W),
            mode=resampling,
            align_corners=resampling in align_corner_filters,
        )

        # Optionally supersample
        if supersample:
            resized_image = F.interpolate(
                resized_image,
                scale_factor=2,
                mode=resampling,
                align_corners=resampling in align_corner_filters,
            )

        # Transpose back to the original format: BxHxWxC or HxWxC
        if len(image.shape) == 4:
            resized_image = resized_image.permute(0, 2, 3, 1)
        else:
            resized_image = resized_image.squeeze(0).permute(1, 2, 0)

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) != len(resized_image.shape):
                raise ValueError(
                    "Mask tensor should have the same dimensions as the image tensor"
                )
            resized_image = resized_image * mask

        return (resized_image,)

```
