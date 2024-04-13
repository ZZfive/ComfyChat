# GetImageRangeFromBatch
## Documentation
- Class name: `GetImageRangeFromBatch`
- Category: `KJNodes`
- Output node: `False`

This node extracts a specific range of images from a batch based on the provided start index and the number of frames. It is designed to facilitate the selection and manipulation of subsets of images within a larger collection.
## Input types
### Required
- **`images`**
    - The batch of images from which a range will be selected. This parameter is crucial for determining the subset of images to be extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start_index`**
    - The index within the batch from which to start the selection. A special value of -1 indicates starting from the end backwards. This parameter determines the starting point of the image range to be extracted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`num_frames`**
    - The number of images to be selected from the start index. This defines the size of the image range to be extracted from the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The selected range of images from the batch.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImageRangeFromBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imagesfrombatch"
    CATEGORY = "KJNodes"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
        },
    } 
    
    def imagesfrombatch(self, images, start_index, num_frames):
        if start_index == -1:
            start_index = len(images) - num_frames
        if start_index < 0 or start_index >= len(images):
            raise ValueError("GetImageRangeFromBatch: Start index is out of range")
        end_index = start_index + num_frames
        if end_index > len(images):
            raise ValueError("GetImageRangeFromBatch: End index is out of range")
        chosen_images = images[start_index:end_index]
        return (chosen_images, )

```
