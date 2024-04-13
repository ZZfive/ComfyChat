# OffsetMaskByNormalizedAmplitude
## Documentation
- Class name: `OffsetMaskByNormalizedAmplitude`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to modify a given mask by applying an offset and rotation based on normalized amplitude values. It ensures that the amplitude values are within a specific range and uses these values to determine the degree of rotation and the magnitude of the offset in both the x and y directions, effectively transforming the mask in a way that is proportional to the provided amplitude.
## Input types
### Required
- **`normalized_amp`**
    - An array of normalized amplitude values, each within the range [0, 1], used to determine the degree of offset and rotation to apply to the mask.
    - Comfy dtype: `NORMALIZED_AMPLITUDE`
    - Python dtype: `numpy.ndarray`
- **`mask`**
    - The mask tensor to be transformed. It serves as the base for applying the offset and rotation transformations.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`x`**
    - The horizontal offset multiplier. It influences the magnitude of horizontal shift applied to the mask.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`y`**
    - The vertical offset multiplier. It influences the magnitude of vertical shift applied to the mask.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`rotate`**
    - A boolean flag indicating whether rotation should be applied to the mask based on the normalized amplitude.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`angle_multiplier`**
    - A multiplier for the rotation angle, which is determined by the normalized amplitude. It affects the degree of rotation applied to the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The transformed mask after applying the specified offsets and rotations, based on the normalized amplitude values.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class OffsetMaskByNormalizedAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "rotate": ("BOOLEAN", { "default": False }),
                "angle_multiplier": ("FLOAT", { "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "display": "number" }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "KJNodes/masking"

    def offset(self, mask, x, y, angle_multiplier, rotate, normalized_amp):

         # Ensure normalized_amp is an array and within the range [0, 1]
        offsetmask = mask.clone()
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
       
        batch_size, height, width = mask.shape

        if rotate:
            for i in range(batch_size):
                rotation_amp = int(normalized_amp[i] * (360 * angle_multiplier))
                rotation_angle = rotation_amp
                offsetmask[i] = TF.rotate(offsetmask[i].unsqueeze(0), rotation_angle).squeeze(0)
        if x != 0 or y != 0:
            for i in range(batch_size):
                offset_amp = normalized_amp[i] * 10
                shift_x = min(x*offset_amp, width-1)
                shift_y = min(y*offset_amp, height-1)
                if shift_x != 0:
                    offsetmask[i] = torch.roll(offsetmask[i], shifts=int(shift_x), dims=1)
                if shift_y != 0:
                    offsetmask[i] = torch.roll(offsetmask[i], shifts=int(shift_y), dims=0)
        
        return offsetmask,

```
