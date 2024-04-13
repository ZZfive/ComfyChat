# 🔧 Image Composite From Mask Batch
## Documentation
- Class name: `ImageCompositeFromMaskBatch+`
- Category: `essentials`
- Output node: `False`

This node is designed to composite two images based on a given mask, blending portions of each image to create a new image. It adjusts the dimensions of the input images and the mask to ensure compatibility before performing the compositing operation.
## Input types
### Required
- **`image_from`**
    - The source image from which pixels are taken when the corresponding mask value is closer to 0. It plays a crucial role in determining the final appearance of the composited image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_to`**
    - The target image to which pixels are added when the corresponding mask value is closer to 1. This image contributes to the final composited output, especially in areas specified by the mask.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A binary or grayscale mask that determines how pixels from the 'image_from' and 'image_to' are blended together. The mask's values guide the compositing process, influencing the output image's appearance.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after compositing 'image_from' and 'image_to' based on the 'mask'. It represents a blend of both input images as dictated by the mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageCompositeFromMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "mask": ("MASK", )
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image_from, image_to, mask):
        frames = mask.shape[0]

        if image_from.shape[1] != image_to.shape[1] or image_from.shape[2] != image_to.shape[2]:
            image_to = p(image_to)
            image_to = comfy.utils.common_upscale(image_to, image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center')
            image_to = pb(image_to)
               
        if frames < image_from.shape[0]:
            image_from = image_from[:frames]
        elif frames > image_from.shape[0]:
            image_from = torch.cat((image_from, image_from[-1].unsqueeze(0).repeat(frames-image_from.shape[0], 1, 1, 1)), dim=0)
        
        mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        if image_from.shape[1] != mask.shape[1] or image_from.shape[2] != mask.shape[2]:
            mask = p(mask)
            mask = comfy.utils.common_upscale(mask, image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center')
            mask = pb(mask)

        out = mask * image_to + (1 - mask) * image_from

        return (out, )

```
