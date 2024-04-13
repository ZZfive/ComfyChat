# RemapFromQuadrilateral (homography)
## Documentation
- Class name: `RemapFromQuadrilateral (homography)`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapFromQuadrilateral node is designed to transform an image by remapping it from a quadrilateral shape to a rectangular shape using homography. This node is particularly useful for correcting perspective distortions in images, allowing for the adjustment of images to a standard orientation and size.
## Input types
### Required
- **`src_mask_with_4_points`**
    - Specifies the source mask with four points defining the quadrilateral to be transformed. This input is crucial for determining the transformation matrix for the remapping process.
    - Comfy dtype: `MASK`
    - Python dtype: `ndarray`
- **`width`**
    - Defines the width of the output image after the remapping process. This parameter allows for the resizing of the transformed image to a specific dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Defines the height of the output image after the remapping process. Similar to the width parameter, it allows for the resizing of the transformed image to a desired dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output of the node after applying the homography transformation to remap the image from a quadrilateral to a rectangular shape.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapFromQuadrilateral(RemapBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "src_mask_with_4_points": ("MASK",),
            #"mode": (s.modes_list, {"default": s.modes_list[0]}),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    @staticmethod
    def homography(*args):
        ret, mask, bb = RemapQuadrilateral.homography(*args)
        return ret, mask, None

    def send_remap(self, src_mask_with_4_points, width, height):
        from .utils.remaps import remap_from_quadrilateral
        remap_data = {
            "func": remap_from_quadrilateral,
            "xargs": [tensor2opencv(src_mask_with_4_points, 1), width, height],
            "dims": (width, height),  # seems kinda redundant, not sure if should refactor
            "custom": RemapFromQuadrilateral.homography
        }
        return (remap_data,)

```
