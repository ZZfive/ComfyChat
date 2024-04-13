# 🔧 Image Seam Carving
## Documentation
- Class name: `ImageSeamCarving+`
- Category: `essentials`
- Output node: `False`

This node specializes in dynamically resizing images through the process of seam carving, a technique that selectively removes or adds pixels in paths (or seams) from an image to reduce or increase its size without distorting the content. It supports custom energy functions, seam ordering, and the use of masks to protect or target specific areas during the resizing process.
## Input types
### Required
- **`image`**
    - The input image to be resized using seam carving. It serves as the primary subject for the dynamic resizing process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - The target width for the resized image. This parameter determines the new width dimension the image will be resized to.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The target height for the resized image. This parameter determines the new height dimension the image will be resized to.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`energy`**
    - Specifies the energy function to use for identifying seams. It influences how seams are chosen for removal or addition based on pixel importance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`order`**
    - Determines the order in which seams are processed (e.g., width-first or height-first), affecting the resizing strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`keep_mask`**
    - A mask indicating areas of the image to preserve during the seam carving process. It helps in protecting specific content from being altered.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
- **`drop_mask`**
    - A mask indicating areas of the image that can be preferentially removed during the seam carving process. It targets specific content for alteration.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resized image after applying the seam carving process. It reflects the dynamic resizing outcome, adhering to the specified width, height, and other parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageSeamCarving:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "energy": (["backward", "forward"],),
                "order": (["width-first", "height-first"],),
            },
            "optional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials"
    FUNCTION = "execute"

    def execute(self, image, width, height, energy, order, keep_mask=None, drop_mask=None):
        try:
            from .carve import seam_carving
        except ImportError as e:
            raise Exception(e)

        img = p(image)

        if keep_mask is not None:
            #keep_mask = keep_mask.reshape((-1, 1, keep_mask.shape[-2], keep_mask.shape[-1])).movedim(1, -1)
            keep_mask = p(keep_mask.unsqueeze(-1))

            if keep_mask.shape[2] != img.shape[2] or keep_mask.shape[3] != img.shape[3]:
                keep_mask = F.interpolate(keep_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")
        if drop_mask is not None:
            drop_mask = p(drop_mask.unsqueeze(-1))

            if drop_mask.shape[2] != img.shape[2] or drop_mask.shape[3] != img.shape[3]:
                drop_mask = F.interpolate(drop_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")

        out = []
        for i in range(img.shape[0]):
            resized = seam_carving(
                T.ToPILImage()(img[i]),
                size=(width, height), 
                energy_mode=energy,
                order=order,
                keep_mask=T.ToPILImage()(keep_mask[i]) if keep_mask is not None else None,
                drop_mask=T.ToPILImage()(drop_mask[i]) if drop_mask is not None else None,
            )
            out.append(T.ToTensor()(resized))
        
        out = torch.stack(out)
        out = pb(out)

        return(out, )

```
