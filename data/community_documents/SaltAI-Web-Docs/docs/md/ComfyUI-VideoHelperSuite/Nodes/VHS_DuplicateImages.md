# Duplicate Image Batch 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_DuplicateImages`
- Category: `Video Helper Suite 🎥🅥🅗🅢/image`
- Output node: `False`

This node is designed to duplicate a batch of images a specified number of times. It is useful for creating larger datasets from a smaller number of images or for repeating visual elements in video or image processing tasks.
## Input types
### Required
- **`images`**
    - The batch of images to be duplicated. This parameter is crucial for defining the input data that will be processed and replicated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Tensor`
- **`multiply_by`**
    - Specifies the number of times the input images should be duplicated. This parameter directly influences the size of the output dataset by determining how many copies of each image will be created.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of duplicated images, with the total number of images being the original count multiplied by the 'multiply_by' factor.
    - Python dtype: `Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of images in the output batch, calculated as the original count multiplied by the 'multiply_by' factor.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DuplicateImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, images: Tensor, multiply_by: int):
        full_images = []
        for n in range(0, multiply_by):
            full_images.append(images)
        new_images = torch.cat(full_images, dim=0)
        return (new_images, new_images.size(0),)

```
