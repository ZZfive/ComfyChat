# Select Every Nth Image 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthImage`
- Category: `Video Helper Suite 🎥🅥🅗🅢/image`
- Output node: `False`

This node is designed to filter a batch of images by selecting every Nth image from the batch. It's useful for thinning out image sequences or datasets to reduce processing load or to create a subset of images for specific purposes.
## Input types
### Required
- **`images`**
    - The batch of images to be filtered. This parameter is crucial for determining which images will be included in the output based on the selection criteria.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Tensor`
- **`select_every_nth`**
    - Determines the interval at which images are selected from the batch. For example, a value of 2 would mean every second image is selected. This parameter directly influences the size of the output batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The filtered batch of images, containing only every Nth image based on the selection criteria.
    - Python dtype: `Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of images selected and included in the output batch.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "select_images"

    def select_images(self, images: Tensor, select_every_nth: int):
        sub_images = images[0::select_every_nth]
        return (sub_images, sub_images.size(0))

```
