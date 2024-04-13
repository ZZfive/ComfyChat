# OffsetMask
## Documentation
- Class name: `OffsetMask`
- Category: `KJNodes/masking`
- Output node: `False`

The OffsetMask node is designed to manipulate the spatial positioning of mask elements within a batch. It enables the creation of variations by duplicating the mask and applying transformations such as rotation and shifting, based on specified parameters. This functionality is crucial for augmenting data or creating more diverse datasets by introducing spatial variability to the masks.
## Input types
### Required
- **`mask`**
    - The mask to be transformed. It serves as the base for creating duplicates and applying spatial transformations, thereby enabling the generation of varied mask positions.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`x`**
    - The horizontal shift factor. It determines the extent of horizontal displacement applied to the mask, contributing to the spatial variation of the transformed masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The vertical shift factor. It influences the vertical displacement of the mask, playing a key role in altering its spatial positioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - The angle factor for rotation. It specifies the degree of rotation to be applied to each mask duplicate, introducing rotational variability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`duplication_factor`**
    - Specifies the number of times the mask is duplicated before transformations. It controls the volume of generated mask variations, directly impacting dataset diversity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roll`**
    - A boolean indicating whether to apply rolling transformations. It toggles the application of rolling shifts to the mask, adding another dimension of spatial manipulation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`incremental`**
    - A boolean flag that determines whether transformations are applied incrementally. It affects the progression of spatial transformations, potentially leading to more nuanced variations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`padding_mode`**
    - Defines the padding strategy used during transformations. It affects how the mask is handled at the edges during rotation and shifting, influencing the final appearance of the transformed masks.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The transformed mask after applying duplication, rotation, and shifting based on the input parameters. It represents the output of spatial manipulations, ready for further processing or use.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class OffsetMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": MAX_RESOLUTION, "step": 1, "display": "number" }),
                "angle": ("INT", { "default": 0, "min": -360, "max": 360, "step": 1, "display": "number" }),
                "duplication_factor": ("INT", { "default": 1, "min": 1, "max": 1000, "step": 1, "display": "number" }),
                "roll": ("BOOLEAN", { "default": False }),
                "incremental": ("BOOLEAN", { "default": False }),
                "padding_mode": (
            [   
                'empty',
                'border',
                'reflection',
                
            ], {
               "default": 'empty'
            }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "KJNodes/masking"

    def offset(self, mask, x, y, angle, roll=False, incremental=False, duplication_factor=1, padding_mode="empty"):
        # Create duplicates of the mask batch
        mask = mask.repeat(duplication_factor, 1, 1).clone()

        batch_size, height, width = mask.shape

        if angle != 0 and incremental:
            for i in range(batch_size):
                rotation_angle = angle * (i+1)
                mask[i] = TF.rotate(mask[i].unsqueeze(0), rotation_angle).squeeze(0)
        elif angle > 0:
            for i in range(batch_size):
                mask[i] = TF.rotate(mask[i].unsqueeze(0), angle).squeeze(0)

        if roll:
            if incremental:
                for i in range(batch_size):
                    shift_x = min(x*(i+1), width-1)
                    shift_y = min(y*(i+1), height-1)
                    if shift_x != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_x, dims=1)
                    if shift_y != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_y, dims=0)
            else:
                shift_x = min(x, width-1)
                shift_y = min(y, height-1)
                if shift_x != 0:
                    mask = torch.roll(mask, shifts=shift_x, dims=2)
                if shift_y != 0:
                    mask = torch.roll(mask, shifts=shift_y, dims=1)
        else:
            
            for i in range(batch_size):
                if incremental:
                    temp_x = min(x * (i+1), width-1)
                    temp_y = min(y * (i+1), height-1)
                else:
                    temp_x = min(x, width-1)
                    temp_y = min(y, height-1)
                if temp_x > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((height, temp_x)), mask[i, :, :-temp_x]], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = pad(mask[i, :, :-temp_x], (0, temp_x), mode=padding_mode)
                elif temp_x < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :, :temp_x], torch.zeros((height, -temp_x))], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = pad(mask[i, :, -temp_x:], (temp_x, 0), mode=padding_mode)

                if temp_y > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((temp_y, width)), mask[i, :-temp_y, :]], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = pad(mask[i, :-temp_y, :], (0, temp_y), mode=padding_mode)
                elif temp_y < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :temp_y, :], torch.zeros((-temp_y, width))], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = pad(mask[i, -temp_y:, :], (temp_y, 0), mode=padding_mode)
           
        return mask,

```
