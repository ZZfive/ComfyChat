# 🔧 Image Apply LUT
## Documentation
- Class name: `ImageApplyLUT+`
- Category: `essentials`
- Output node: `False`

The ImageApplyLUT+ node applies a Look-Up Table (LUT) to images, optionally adjusting for non-default color domains and applying logarithmic color space transformations. It supports clipping LUT values, scaling image intensity based on the LUT's domain, and blending the transformed image with the original based on a specified strength.
## Input types
### Required
- **`image`**
    - The input image(s) to which the LUT will be applied. This can involve color space transformations and intensity scaling.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`lut_file`**
    - The file name of the Look-Up Table (LUT) to be applied to the input image(s).
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`log_colorspace`**
    - A flag indicating whether to apply logarithmic color space transformation to the image(s) before and after applying the LUT.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`clip_values`**
    - A flag indicating whether to clip the LUT values to their domain, ensuring that the output values do not exceed the specified range.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`strength`**
    - A value between 0 and 1 indicating the blend strength of the LUT-applied image with the original image. A strength of 1 means full application of the LUT, while 0 means no change.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image(s) after applying the LUT, with optional clipping, color space transformation, and blending based on the specified strength.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": ([f for f in os.listdir(LUTS_DIR) if f.endswith('.cube')], ),
                "log_colorspace": ("BOOLEAN", { "default": False }),
                "clip_values": ("BOOLEAN", { "default": False }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, log_colorspace, clip_values, strength):
        from colour.io.luts.iridas_cube import read_LUT_IridasCube
        
        lut = read_LUT_IridasCube(os.path.join(LUTS_DIR, lut_file))
        lut.name = lut_file

        if clip_values:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # 3x1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        out = []
        for img in image: # TODO: is this more resource efficient? should we use a batch instead?
            lut_img = img.numpy().copy()

            is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut.domain[1] - lut.domain[0]
                lut_img = lut_img * dom_scale + lut.domain[0]
            if log_colorspace:
                lut_img = lut_img ** (1/2.2)
            lut_img = lut.apply(lut_img)
            if log_colorspace:
                lut_img = lut_img ** (2.2)
            if is_non_default_domain:
                lut_img = (lut_img - lut.domain[0]) / dom_scale

            lut_img = torch.from_numpy(lut_img)
            if strength < 1.0:
                lut_img = strength * lut_img + (1 - strength) * img
            out.append(lut_img)
        
        out = torch.stack(out)

        return (out, )

```
