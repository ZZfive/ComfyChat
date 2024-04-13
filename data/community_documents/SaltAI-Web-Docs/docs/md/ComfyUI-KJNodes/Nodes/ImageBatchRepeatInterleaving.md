# ImageBatchRepeatInterleaving
## Documentation
- Class name: `ImageBatchRepeatInterleaving`
- Category: `KJNodes`
- Output node: `False`

This node specializes in duplicating images within a batch, allowing each image to be repeated a specified number of times. It enhances the flexibility of image batch manipulation by enabling the creation of larger batches from existing images through repetition.
## Input types
### Required
- **`images`**
    - The collection of images to be repeated. This parameter is crucial for determining the content of the output batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`repeats`**
    - Specifies the number of times each image in the input batch should be repeated. This parameter directly influences the size of the output batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A batch of images where each input image has been repeated according to the specified 'repeats' parameter.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchRepeatInterleaving:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
        },
    } 
    
    def repeat(self, images, repeats):
       
        repeated_images = torch.repeat_interleave(images, repeats=repeats, dim=0)
        return (repeated_images, )

```
