# 🔧 Mask From Color
## Documentation
- Class name: `MaskFromColor+`
- Category: `essentials`
- Output node: `False`

The `MaskFromColor` node is designed to generate a mask from an image based on specific color values and a threshold. It allows for the creation of masks by identifying areas within an image that match the specified color within a certain tolerance, enabling selective editing or processing of those areas.
## Input types
### Required
- **`image`**
    - The input image from which the mask will be generated. The mask creation process involves identifying areas that match the specified color values.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`red`**
    - The red component of the target color. This value, along with green and blue, defines the color to match in the image for mask generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`green`**
    - The green component of the target color. It specifies the green intensity to be matched in the image for creating the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blue`**
    - The blue component of the target color. It defines the blue intensity in the image that will be used for mask creation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - The tolerance threshold for color matching. It allows for flexibility in matching the specified color within the image, accommodating slight variations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask generated from the input image, based on the specified color values and threshold. This mask can be used for further image processing tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskFromColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "red": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "green": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "blue": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, red, green, blue, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = torch.tensor([red, green, blue])
        lower_bound = (color - threshold).clamp(min=0)
        upper_bound = (color + threshold).clamp(max=255)
        lower_bound = lower_bound.view(1, 1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 1, 3)
        mask = (temp >= lower_bound) & (temp <= upper_bound)
        mask = mask.all(dim=-1)
        mask = mask.float()

        return (mask, )

```
