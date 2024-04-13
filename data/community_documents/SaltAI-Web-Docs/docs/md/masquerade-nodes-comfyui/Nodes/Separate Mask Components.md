# Separate Mask Components
## Documentation
- Class name: `Separate Mask Components`
- Category: `Masquerade Nodes`
- Output node: `False`

This node is designed to process a given mask tensor, identifying and separating its distinct components based on connectivity. It effectively segments the mask into individual, non-overlapping regions, each corresponding to a connected component within the original mask.
## Input types
### Required
- **`mask`**
    - The mask tensor to be segmented into distinct components. It plays a crucial role in determining the segmentation outcome by identifying connected regions within the mask.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor containing the separated mask components, where each component is represented as a distinct region in the tensor.
    - Python dtype: `torch.Tensor`
- **`mask_mapping`**
    - Comfy dtype: `MASK_MAPPING`
    - A tensor mapping each separated component to its original index in the batch, facilitating tracking of component origins.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SeparateMaskComponents:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK_MAPPING")
    FUNCTION = "separate"

    CATEGORY = "Masquerade Nodes"

    def separate(self, mask):
        mask = tensor2mask(mask)

        thresholded = torch.gt(mask,0).unsqueeze(1)
        B, H, W = mask.shape
        components = torch.arange(B * H * W, device=mask.device, dtype=mask.dtype).reshape(B, 1, H, W) + 1
        components[~thresholded] = 0

        while True:
            previous_components = components
            components = torch.nn.functional.max_pool2d(components, kernel_size=3, stride=1, padding=1)
            components[~thresholded] = 0
            if torch.equal(previous_components, components):
                break

        components = components.reshape(B, H, W)
        segments = torch.unique(components)
        result = torch.zeros([len(segments) - 1, H, W])
        index = 0
        mapping = torch.zeros([len(segments) - 1], device=mask.device, dtype=torch.int)
        for i in range(len(segments)):
            segment = segments[i].item()
            if segment == 0:
                continue
            image_index = int((segment - 1) // (H * W))
            segment_mask = (components[image_index,:,:] == segment)
            result[index][segment_mask] = mask[image_index][segment_mask]
            mapping[index] = image_index
            index += 1

        return (result,mapping,)

```
