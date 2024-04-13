# 🔧 Image From Batch
## Documentation
- Class name: `ImageFromBatch+`
- Category: `essentials`
- Output node: `False`

This node is designed to extract a specific segment from a batch of images based on the provided start index and length. It allows for flexible manipulation of image batches by enabling selective retrieval of images, which can be particularly useful in processing or analyzing subsets of image collections.
## Input types
### Required
- **`image`**
    - The batch of images from which a segment is to be extracted. This parameter is crucial for specifying the source of the images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start`**
    - The starting index from which the image segment extraction should begin. It determines the initial position in the batch from which images will be selected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`length`**
    - The number of images to extract from the specified starting index. This parameter defines the size of the segment to be retrieved from the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The extracted segment of images from the specified batch. This output is a subset of the original batch, determined by the 'start' and 'length' parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, start, length):
        if length<0:
            length = image.shape[0]
        start = min(start, image.shape[0]-1)
        length = min(image.shape[0]-start, length)
        return (image[start:start + length], )

```
