# ImageFromBatch
## Documentation
- Class name: `ImageFromBatch`
- Category: `image/batch`
- Output node: `False`

The ImageFromBatch node is designed for extracting a specific segment of images from a batch based on the provided index and length. It allows for more granular control over the batched images, enabling operations on individual or subsets of images within a larger batch.
## Input types
### Required
- **`image`**
    - The batch of images from which a segment will be extracted. This parameter is crucial for specifying the source batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`batch_index`**
    - The starting index within the batch from which the extraction begins. It determines the initial position of the segment to be extracted from the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`length`**
    - The number of images to extract from the batch starting from the batch_index. This parameter defines the size of the segment to be extracted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The extracted segment of images from the specified batch. This output represents a subset of the original batch, determined by the batch_index and length parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                              "length": ("INT", {"default": 1, "min": 1, "max": 64}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frombatch"

    CATEGORY = "image/batch"

    def frombatch(self, image, batch_index, length):
        s_in = image
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return (s,)

```
