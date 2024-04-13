# Image Compare (mtb)
## Documentation
- Class name: `Image Compare (mtb)`
- Category: `mtb/image`
- Output node: `False`

This node compares two images based on a specified mode and returns a difference image, allowing for visual comparison between the two inputs.
## Input types
### Required
- **`imageA`**
    - The first image to be compared. It plays a crucial role in the comparison process as the base image against which the second image is compared.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`imageB`**
    - The second image to be compared. It is compared against the first image to identify differences or similarities.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the method of comparison (checkerboard, diff, blend). This affects how the differences between the two images are visualized.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after comparing the two input images according to the specified mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageCompare_:
    """Compare two images and return a difference image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "imageB": ("IMAGE",),
                "mode": (
                    ["checkerboard", "diff", "blend"],
                    {"default": "checkerboard"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare"
    CATEGORY = "mtb/image"

    def compare(self, imageA: torch.Tensor, imageB: torch.Tensor, mode):
        imageA = imageA.numpy()
        imageB = imageB.numpy()

        imageA = imageA.squeeze()
        imageB = imageB.squeeze()

        image = compare_images(imageA, imageB, method=mode)

        image = np.expand_dims(image, axis=0)
        return (torch.from_numpy(image),)

```
