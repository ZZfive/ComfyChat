# Batch Make (mtb)
## Documentation
- Class name: `Batch Make (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The Batch Make node is designed to create a batch of identical images from a single input image, effectively duplicating the input image a specified number of times to form a batch. This functionality is crucial for operations that require batch processing of images, allowing for the efficient handling of multiple instances of the same image in parallel.
## Input types
### Required
- **`image`**
    - The input image to be duplicated into a batch. It serves as the base image from which the batch is generated, playing a pivotal role in the batch creation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`count`**
    - Specifies the number of times the input image should be duplicated to form the batch. This parameter directly influences the size of the resulting batch, thereby affecting the node's output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images, each identical to the input image, created by duplicating the input image a specified number of times.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchMake:
    """Simply duplicates the input frame as a batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_batch"
    CATEGORY = "mtb/batch"

    def generate_batch(self, image: torch.Tensor, count):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return (image.repeat(count, 1, 1, 1),)

```
