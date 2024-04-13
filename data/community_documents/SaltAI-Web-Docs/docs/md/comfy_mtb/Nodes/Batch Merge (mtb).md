# Batch Merge (mtb)
## Documentation
- Class name: `Batch Merge (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The BatchMerge node is designed to merge multiple image batches of varying frame counts into a single batch. It adjusts the frame count of each batch to match the maximum frame count found among them, using a specified filling strategy, and then merges them using a specified fusion mode.
## Input types
### Required
- **`fusion_mode`**
    - Specifies the method of merging the adjusted image batches. It can be 'add', 'multiply', or 'average', affecting the final merged image's appearance based on the selected operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fill`**
    - Determines the strategy for filling frames in batches with fewer frames than the maximum. 'head' fills from the beginning, and 'tail' fills from the end, influencing the content and appearance of the adjusted batches.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The merged image batch resulting from combining the adjusted image batches according to the specified fusion mode and fill strategy.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchMerge:
    """Merges multiple image batches with different frame counts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fusion_mode": (
                    ["add", "multiply", "average"],
                    {"default": "average"},
                ),
                "fill": (["head", "tail"], {"default": "tail"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_batches"
    CATEGORY = "mtb/batch"

    def merge_batches(self, fusion_mode, fill, **kwargs):
        images = kwargs.values()
        max_frames = max(img.shape[0] for img in images)

        adjusted_images = []
        for img in images:
            frame_count = img.shape[0]
            if frame_count < max_frames:
                fill_frame = img[0] if fill == "head" else img[-1]
                fill_frames = fill_frame.repeat(
                    max_frames - frame_count, 1, 1, 1
                )
                adjusted_batch = (
                    torch.cat((fill_frames, img), dim=0)
                    if fill == "head"
                    else torch.cat((img, fill_frames), dim=0)
                )
            else:
                adjusted_batch = img
            adjusted_images.append(adjusted_batch)

        # Merge the adjusted batches
        merged_image = None
        for img in adjusted_images:
            if merged_image is None:
                merged_image = img
            else:
                if fusion_mode == "add":
                    merged_image += img
                elif fusion_mode == "multiply":
                    merged_image *= img
                elif fusion_mode == "average":
                    merged_image = (merged_image + img) / 2

        return (merged_image,)

```
