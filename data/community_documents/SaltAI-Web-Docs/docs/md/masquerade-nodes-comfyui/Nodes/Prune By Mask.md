# Prune By Mask
## Documentation
- Class name: `Prune By Mask`
- Category: `Masquerade Nodes`
- Output node: `False`

The Prune By Mask node is designed to filter out parts of an image based on a mask, retaining only those regions where the mask indicates presence (i.e., mask values are above a certain threshold).
## Input types
### Required
- **`image`**
    - The image input is the primary image that will be pruned based on the mask. It plays a crucial role in determining the output as it is the source from which regions are either retained or discarded.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask input specifies which parts of the image to keep. Regions with mask values above a specified threshold are retained, making the mask pivotal in shaping the final pruned image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a pruned version of the input image, containing only those regions specified by the mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PruneByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prune"

    CATEGORY = "Masquerade Nodes"

    def prune(self, image, mask):
        mask = tensor2mask(mask)

        mean = torch.mean(torch.mean(mask,dim=2),dim=1)
        return (image[mean >= 0.5],)

```
