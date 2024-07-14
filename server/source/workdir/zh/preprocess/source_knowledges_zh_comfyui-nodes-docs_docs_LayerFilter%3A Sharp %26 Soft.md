# Documentation
- Class name: SharpAndSoft
- Category: 😺dzNodes/LayerFilter
- Output node: False
- Repo Ref: https://github.com/chflame163/ComfyUI_LayerStyle

为图像增强细节或抹平细节。

# Input types

## Required

- images
    - 图片。
    - Comfy dtype: IMAGE
    - Python dtype: torch.Tensor

- enhance
    - 提供四个预设档位，分别是very sharp、sharp、soft和very soft。选None则不做任何处理。
    - Comfy dtype: STRING_ONEOF
    - Python dtype: str
    - Options: 
        - very sharp
        - sharp
        - soft
        - very soft
        - None

# Output types

- image
    - 图片。
    - Comfy dtype: IMAGE
    - Python dtype: torch.Tensor

# Usage tips
- Infra type: CPU

# Source code
```python
class SharpAndSoft:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        enhance_list = ['very sharp', 'sharp', 'soft', 'very soft', 'None']

        return {
            "required": {
                "images": ("IMAGE",),
                "enhance": (enhance_list, ),

            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'sharp_and_soft'
    CATEGORY = '😺dzNodes/LayerFilter'

    def sharp_and_soft(self, images, enhance, ):

        if enhance == 'very sharp':
            filter_radius = 1
            denoise = 0.6
            detail_mult = 2.8
        if enhance == 'sharp':
            filter_radius = 3
            denoise = 0.12
            detail_mult = 1.8
        if enhance == 'soft':
            filter_radius = 8
            denoise = 0.08
            detail_mult = 0.5
        if enhance == 'very soft':
            filter_radius = 15
            denoise = 0.06
            detail_mult = 0.01
        else:
            return (images,)

        d = int(filter_radius * 2) + 1
        s = 0.02
        n = denoise / 10
        dup = copy.deepcopy(images.cpu().numpy())

        for index, image in enumerate(dup):
            imgB = image
            if denoise > 0.0:
                imgB = cv2.bilateralFilter(image, d, n, d)
            imgG = np.clip(guidedFilter(image, image, d, s), 0.001, 1)
            details = (imgB / imgG - 1) * detail_mult + 1
            dup[index] = np.clip(details * imgG - imgB + image, 0, 1)

        log(f"{NODE_NAME} Processed {dup.shape[0]} image(s).", message_type='finish')
        return (torch.from_numpy(dup),)

```