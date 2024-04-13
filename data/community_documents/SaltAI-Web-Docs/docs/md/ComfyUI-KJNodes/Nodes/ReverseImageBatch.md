# ReverseImageBatch
## Documentation
- Class name: `ReverseImageBatch`
- Category: `KJNodes`
- Output node: `False`

The ReverseImageBatch node is designed to reverse the order of images in a batch. It provides a simple yet effective way to manipulate the sequence of images, making it useful for tasks that require the inversion of image order.
## Input types
### Required
- **`images`**
    - The 'images' parameter represents the batch of images to be reversed. It plays a crucial role in determining the output, as the node will reverse the order of these images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images with their order reversed from the input batch.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ReverseImageBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverseimagebatch"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
        },
    } 
    
    def reverseimagebatch(self, images):
        reversed_images = torch.flip(images, [0])
        return (reversed_images, )

```
