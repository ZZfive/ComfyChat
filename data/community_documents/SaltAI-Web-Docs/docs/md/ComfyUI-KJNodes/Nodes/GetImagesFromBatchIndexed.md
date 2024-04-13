# GetImagesFromBatchIndexed
## Documentation
- Class name: `GetImagesFromBatchIndexed`
- Category: `KJNodes`
- Output node: `False`

This node is designed to extract specific images from a batch based on provided indices. It parses a string of indices, converts them into a tensor, and selects the corresponding images from the batch.
## Input types
### Required
- **`images`**
    - The batch of images from which specific images are to be selected. It plays a crucial role in determining the output by providing the source images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`indexes`**
    - A string of comma-separated indices indicating which images to extract from the batch. This parameter directly influences which images are selected and returned.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The selected images from the batch at the specified indices.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImagesFromBatchIndexed:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
        },
    } 
    
    def indexedimagesfrombatch(self, images, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Select the images at the specified indices
        chosen_images = images[indices_tensor]
        
        return (chosen_images,)

```
