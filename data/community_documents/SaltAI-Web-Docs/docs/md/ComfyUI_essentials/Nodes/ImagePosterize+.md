# 🔧 Image Posterize
## Documentation
- Class name: `ImagePosterize+`
- Category: `essentials`
- Output node: `False`

This node applies a posterization effect to an image, reducing its color depth according to a specified threshold. It simplifies the image's color palette, creating a stylized, graphic effect.
## Input types
### Required
- **`image`**
    - The input image to be posterized. This parameter is crucial as it determines the base image that will undergo the posterization process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`threshold`**
    - Defines the threshold for the posterization effect. A lower threshold results in fewer colors and a more pronounced posterization effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the posterization effect. It features reduced color depth, creating a stylized appearance.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImagePosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, threshold):
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        #image = image.mean(dim=3, keepdim=True)
        image = (image > threshold).float()
        image = image.unsqueeze(-1).repeat(1, 1, 1, 3)

        return(image,)

```
