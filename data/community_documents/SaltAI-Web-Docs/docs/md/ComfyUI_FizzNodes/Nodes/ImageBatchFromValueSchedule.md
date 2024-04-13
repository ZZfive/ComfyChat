# Image Batch From Value Schedule 📅🅕🅝
## Documentation
- Class name: `ImageBatchFromValueSchedule`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

The ImageBatchFromValueSchedule node is designed to generate a batch of images based on a value schedule. It utilizes a scheduling mechanism to select and manipulate images from an input batch according to specified values, facilitating dynamic image generation and manipulation within a batch processing context.
## Input types
### Required
- **`images`**
    - The images input allows for specifying the batch of images to be manipulated according to the value schedule. It is essential for determining the source images for the scheduling and manipulation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`values`**
    - The values input specifies the scheduling information or parameters that guide the selection and manipulation of images in the batch. It plays a key role in defining the behavior and output of the node.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images that have been selected and potentially manipulated based on the provided value schedule. This allows for dynamic and scheduled image generation within a batch.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchFromValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "values": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 1.0, "label": "values" }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, images, values):
        values = [values] * n if isinstance(values, float) else values
        min_value, max_value = min(values), max(values)
        i = [(x - min_value) / (max_value - min_value) * (images.shape[0] - 1) for x in values]
        return (images[i], )

```
