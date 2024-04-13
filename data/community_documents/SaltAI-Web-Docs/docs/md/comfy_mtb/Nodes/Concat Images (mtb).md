# Concat Images (mtb)
## Documentation
- Class name: `Concat Images (mtb)`
- Category: `mtb/image`
- Output node: `False`

The Concat Images node is designed to merge multiple image tensors into a single batch tensor. It dynamically adjusts the batch size based on the input tensors, facilitating the processing of images in bulk while maintaining their individual characteristics.
## Input types
### Required
- **`reverse`**
    - Determines the order in which images are concatenated. When set to True, images are concatenated in reverse order, affecting the final batch's arrangement.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single tensor that combines all input image tensors into one, with an updated batch size to reflect the total number of images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ConcatImages:
    """Add images to batch"""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate_tensors"
    CATEGORY = "mtb/image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"reverse": ("BOOLEAN", {"default": False})},
        }

    def concatenate_tensors(self, reverse, **kwargs):
        tensors = tuple(kwargs.values())
        batch_sizes = [tensor.size(0) for tensor in tensors]

        concatenated = torch.cat(tensors, dim=0)

        # Update the batch size in the concatenated tensor
        concatenated_size = list(concatenated.size())
        concatenated_size[0] = sum(batch_sizes)
        concatenated = concatenated.view(*concatenated_size)

        return (concatenated,)

```
