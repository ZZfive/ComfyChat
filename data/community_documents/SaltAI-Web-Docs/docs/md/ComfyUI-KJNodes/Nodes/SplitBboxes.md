# SplitBboxes
## Documentation
- Class name: `SplitBboxes`
- Category: `KJNodes/masking`
- Output node: `False`

The SplitBboxes node is designed to divide a collection of bounding boxes into two separate lists based on a specified index. This functionality is crucial for scenarios where a dataset of bounding boxes needs to be segmented into different groups for further processing or analysis.
## Input types
### Required
- **`bboxes`**
    - The collection of bounding boxes to be split. This parameter is essential for determining how the bounding boxes are divided into two groups.
    - Comfy dtype: `BBOX`
    - Python dtype: `List[Tuple[int, int, int, int]]`
- **`index`**
    - The index at which the collection of bounding boxes is split. This parameter dictates the point of division, influencing the composition of the resulting two lists of bounding boxes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`bboxes_a`**
    - Comfy dtype: `BBOX`
    - The first list of bounding boxes, containing elements from the start of the original collection up to (but not including) the specified index.
    - Python dtype: `List[Tuple[int, int, int, int]]`
- **`bboxes_b`**
    - Comfy dtype: `BBOX`
    - The second list of bounding boxes, containing elements from the specified index to the end of the original collection.
    - Python dtype: `List[Tuple[int, int, int, int]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SplitBboxes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "index": ("INT", {"default": 0,"min": 0, "max": 99999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("BBOX","BBOX",)
    RETURN_NAMES = ("bboxes_a","bboxes_b",)
    FUNCTION = "splitbbox"

    CATEGORY = "KJNodes/masking"

    def splitbbox(self, bboxes, index):
        bboxes_a = bboxes[:index]  # Sub-list from the start of bboxes up to (but not including) the index
        bboxes_b = bboxes[index:]  # Sub-list from the index to the end of bboxes

        return (bboxes_a, bboxes_b,)

```
