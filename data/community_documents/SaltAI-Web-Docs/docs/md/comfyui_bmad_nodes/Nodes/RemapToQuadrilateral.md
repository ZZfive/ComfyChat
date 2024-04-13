# RemapToQuadrilateral
## Documentation
- Class name: `RemapToQuadrilateral`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to transform an image by remapping it to fit within a specified quadrilateral shape. It leverages various remapping techniques, including homography and custom methods, to adjust the image's perspective and alignment according to the defined quadrilateral boundaries.
## Input types
### Required
- **`dst_mask_with_4_points`**
    - Specifies the destination mask that outlines the quadrilateral shape within which the image is to be remapped. This mask is crucial for determining the transformation's target geometry.
    - Comfy dtype: `MASK`
    - Python dtype: `ndarray`
- **`mode`**
    - Determines the remapping technique to be used, such as homography or custom methods, based on the quadrilateral's properties. The mode influences how the image is transformed to fit the specified shape.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - Contains the data necessary for performing the remapping operation, including the transformation function, arguments, and dimensions of the target quadrilateral.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapQuadrilateral(RemapBase):
    from .utils.remaps import quad_remap_methods_map

    modes_list = list(quad_remap_methods_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "dst_mask_with_4_points": ("MASK",),
            "mode": (s.modes_list, {"default": s.modes_list[0]}),
        }
        }

    @staticmethod
    def homography(custom_data, src, interpolation, mask=None):
        h_matrix, bb = custom_data
        bb_width, bb_height = bb[2] - bb[0], bb[3] - bb[1]
        ret = cv.warpPerspective(src, h_matrix, (bb_width, bb_height), flags=interpolation,
                                 borderMode=cv.BORDER_CONSTANT)
        if mask is not None:
            mask = cv.warpPerspective(mask, h_matrix, (bb_width, bb_height), flags=interpolation,
                                      borderMode=cv.BORDER_CONSTANT)
        return ret, mask, bb

    def send_remap(self, dst_mask_with_4_points, mode):
        from .utils.remaps import remap_quadrilateral
        remap_data = {
            "func": remap_quadrilateral,
            "xargs": [tensor2opencv(dst_mask_with_4_points, 1), mode],
            "dims": RemapBase.get_dims(dst_mask_with_4_points)
        }
        if mode == "HOMOGRAPHY":
            remap_data["custom"] = RemapQuadrilateral.homography
        return (remap_data,)

```
