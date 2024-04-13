# Make Image Batch
## Documentation
- Class name: `Make Image Batch`
- Category: `Masquerade Nodes`
- Output node: `False`

The Make Image Batch node is designed to aggregate multiple image inputs into a single batched image tensor. It allows for the dynamic inclusion of up to six images, concatenating them along a specified dimension to form a unified batch that can be processed collectively.
## Input types
### Required
- **`image1`**
    - The primary image to be included in the batch. It serves as the initial component of the resulting batched image tensor.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`image2`**
    - An optional second image to be appended to the batch. Its inclusion increases the batch size by one.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image3`**
    - An optional third image that can be appended to the batch, further expanding the batch size.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image4`**
    - An optional fourth image that can be appended to the batch, contributing to the overall batch size.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image5`**
    - An optional fifth image that can be appended to the batch, increasing the batch's diversity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image6`**
    - An optional sixth image that can be appended to the batch, maximizing the batch size.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting batched image tensor, comprising the concatenated input images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MakeImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"

    CATEGORY = "Masquerade Nodes"

    def append(self, image1, image2 = None, image3 = None, image4 = None, image5 = None, image6 = None):
        result = image1
        if image2 is not None:
            result = torch.cat((result, image2), 0)
        if image3 is not None:
            result = torch.cat((result, image3), 0)
        if image4 is not None:
            result = torch.cat((result, image4), 0)
        if image5 is not None:
            result = torch.cat((result, image5), 0)
        if image6 is not None:
            result = torch.cat((result, image6), 0)
        return (result,)

```
