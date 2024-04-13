# ColorMatch
## Documentation
- Class name: `ColorMatch`
- Category: `KJNodes/masking`
- Output node: `False`

The ColorMatch node is designed to adjust the color scheme of a target image to match that of a reference image. It utilizes a color matching algorithm to ensure the colors in the target image closely resemble those in the reference image, based on a specified method.
## Input types
### Required
- **`image_ref`**
    - The reference image whose color scheme is to be matched. It plays a crucial role in determining the target image's final color output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_target`**
    - The target image that will be adjusted to match the color scheme of the reference image. This image undergoes transformation based on the reference image's colors.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`method`**
    - Specifies the method to be used for color matching. This affects how the color transformation is applied to the target image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor containing the color-matched target image.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ImageBatch](../../Comfy/Nodes/ImageBatch.md)
    - [VHS_SplitImages](../../ComfyUI-VideoHelperSuite/Nodes/VHS_SplitImages.md)



## Source code
```python
class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                
            },
        }
    
    CATEGORY = "KJNodes/masking"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    
    def colormatch(self, image_ref, image_target, method):
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            out.append(torch.from_numpy(image_result))
        return (torch.stack(out, dim=0).to(torch.float32), )

```
