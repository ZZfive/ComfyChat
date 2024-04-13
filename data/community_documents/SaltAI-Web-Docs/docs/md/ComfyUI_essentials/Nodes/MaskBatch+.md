# 🔧 Mask Batch
## Documentation
- Class name: `MaskBatch+`
- Category: `essentials`
- Output node: `False`

The MaskBatch+ node is designed to combine two mask tensors into a single batched tensor. It ensures that the masks are compatible in terms of dimensions by potentially resizing one of them, and then concatenates them along the batch dimension to produce a unified batch of masks.
## Input types
### Required
- **`mask1`**
    - The first mask tensor to be batched. It serves as the reference for the dimensions of the output batch if resizing is needed.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask2`**
    - The second mask tensor to be batched. It may be resized to match the dimensions of the first mask before batching.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - A batched tensor containing the concatenated masks from the input tensors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask1, mask2):
        if mask1.shape[1:] != mask2.shape[1:]:
            mask2 = F.interpolate(mask2.unsqueeze(1), size=(mask1.shape[1], mask1.shape[2]), mode="bicubic").squeeze(1)
            
        out = torch.cat((mask1, mask2), dim=0)
        return (out,)

```
