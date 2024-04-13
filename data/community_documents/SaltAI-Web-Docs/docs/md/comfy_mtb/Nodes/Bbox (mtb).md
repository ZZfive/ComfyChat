# Bbox (mtb)
## Documentation
- Class name: `Bbox (mtb)`
- Category: `mtb/crop`
- Output node: `False`

The Bbox node is designed to define and manipulate bounding boxes within images. It abstracts the concept of a rectangular area specified by its top-left corner coordinates along with its width and height, facilitating operations like cropping or analyzing specific regions of an image.
## Input types
### Required
- **`x`**
    - The 'x' parameter represents the x-coordinate of the top-left corner of the bounding box. It is crucial for determining the horizontal starting point of the region of interest.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The 'y' parameter signifies the y-coordinate of the top-left corner of the bounding box, essential for identifying the vertical starting point of the region of interest.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The 'width' parameter defines the horizontal extent of the bounding box, determining how wide the area of interest spans.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The 'height' parameter specifies the vertical extent of the bounding box, indicating the height of the area of interest.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`bbox`**
    - Comfy dtype: `BBOX`
    - Returns a tuple representing the bounding box, encapsulating the top-left corner coordinates, width, and height of the specified region.
    - Python dtype: `Tuple[int, int, int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Bbox:
    """The bounding box (BBOX) custom type used by other nodes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "bbox": ("BBOX",),
                "x": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "width": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 256, "max": 10000000, "min": 0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "do_crop"
    CATEGORY = "mtb/crop"

    def do_crop(self, x, y, width, height):  # bbox
        return ((x, y, width, height),)
        # return bbox

```
