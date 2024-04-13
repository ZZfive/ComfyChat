# Pick From Batch (mtb)
## Documentation
- Class name: `Pick From Batch (mtb)`
- Category: `mtb/image utils`
- Output node: `False`

The Pick From Batch node is designed to selectively retrieve a specified number of images from a batch, based on the user's preference for either the beginning or the end of the batch. This functionality allows for flexible manipulation and extraction of subsets from larger collections of images, catering to various processing or analysis needs.
## Input types
### Required
- **`image`**
    - Represents the batch of images from which a subset will be selected. Its role is crucial as it determines the pool of available images for selection.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`from_direction`**
    - Specifies the direction ('start' or 'end') from which images are to be selected, allowing users to tailor the extraction process to their specific requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`count`**
    - Determines the number of images to be extracted from the specified direction, enabling precise control over the size of the resulting subset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a subset of images selected from the specified direction of the original batch, based on the count specified.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PickFromBatch:
    """Pick a specific number of images from a batch, either from the start or end."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "from_direction": (["end", "start"], {"default": "start"}),
                "count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pick_from_batch"
    CATEGORY = "mtb/image utils"

    def pick_from_batch(self, image, from_direction, count):
        batch_size = image.size(0)

        # Limit count to the available number of images in the batch
        count = min(count, batch_size)
        if count < batch_size:
            log.warning(
                f"Requested {count} images, but only {batch_size} are available."
            )

        if from_direction == "end":
            selected_tensors = image[-count:]
        else:
            selected_tensors = image[:count]

        return (selected_tensors,)

```
