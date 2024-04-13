# RepeatImageBatch
## Documentation
- Class name: `RepeatImageBatch`
- Category: `image/batch`
- Output node: `False`

The RepeatImageBatch node is designed to replicate a given image a specified number of times, creating a batch of identical images. This functionality is useful for operations that require multiple instances of the same image, such as batch processing or data augmentation.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the image to be replicated. It is crucial for defining the content that will be duplicated across the batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - The 'amount' parameter specifies the number of times the input image should be replicated. It directly influences the size of the output batch, allowing for flexible batch creation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images, each identical to the input image, replicated according to the specified 'amount'.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RepeatImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"

    CATEGORY = "image/batch"

    def repeat(self, image, amount):
        s = image.repeat((amount, 1,1,1))
        return (s,)

```
