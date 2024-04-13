# ReplaceImagesInBatch
## Documentation
- Class name: `ReplaceImagesInBatch`
- Category: `KJNodes`
- Output node: `False`

This node is designed to replace a subset of images within a batch with another set of images, starting from a specified index. It facilitates the modification of image batches by allowing selective replacement, enhancing flexibility in batch processing tasks.
## Input types
### Required
- **`original_images`**
    - The batch of original images from which a subset will be replaced. This parameter is crucial for defining the base set of images to be modified.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`replacement_images`**
    - The batch of images to replace the original subset with. This parameter is essential for specifying the new images to be inserted into the batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start_index`**
    - The index within the original batch from which the replacement should start. This parameter determines the position at which the new images will be inserted, affecting the outcome of the replacement process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The modified batch of images after the replacement operation, reflecting the changes made by inserting the new images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ReplaceImagesInBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "original_images": ("IMAGE",),
                 "replacement_images": ("IMAGE",),
                 "start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
    } 
    
    def replace(self, original_images, replacement_images, start_index):
        images = None
        if start_index >= len(original_images):
            raise ValueError("GetImageRangeFromBatch: Start index is out of range")
        end_index = start_index + len(replacement_images)
        if end_index > len(original_images):
            raise ValueError("GetImageRangeFromBatch: End index is out of range")
         # Create a copy of the original_images tensor
        original_images_copy = original_images.clone()
        original_images_copy[start_index:end_index] = replacement_images
        images = original_images_copy
        return (images, )

```
