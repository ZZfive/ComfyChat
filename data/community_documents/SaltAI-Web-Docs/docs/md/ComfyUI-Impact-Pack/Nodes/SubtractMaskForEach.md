# Bitwise(SEGS - SEGS)
## Documentation
- Class name: `SubtractMaskForEach`
- Category: `ImpactPack/Operation`
- Output node: `False`

This node is designed to perform a subtraction operation on masks for each frame or segment within a given dataset. It iteratively processes each mask by subtracting it from a subsequent one, effectively isolating changes or movements between frames. The node aims to highlight differences by generating a series of masks that represent the unique aspects of each frame when compared to its neighbors.
## Input types
### Required
- **`base_segs`**
    - A list of base segments. This input is crucial for the node's operation as it determines the sequence of segments to be subtracted from one another, thereby highlighting differences between consecutive frames.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[torch.Tensor]`
- **`mask_segs`**
    - A list of mask segments to be subtracted from the base segments. This input is essential for performing the subtraction operation, highlighting the differences between the base and mask segments.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[torch.Tensor]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output is a list of segments, each representing the unique aspects of a frame by subtracting it from its subsequent frame. This series of segments can be used for further analysis or processing.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SubtractMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "base_segs": ("SEGS",),
                        "mask_segs": ("SEGS",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, base_segs, mask_segs):

        result = []

        for bseg in base_segs[1]:
            cropped_mask1 = bseg.cropped_mask.copy()
            crop_region1 = bseg.crop_region

            for mseg in mask_segs[1]:
                cropped_mask2 = mseg.cropped_mask
                crop_region2 = mseg.crop_region

                # compute the intersection of the two crop regions
                intersect_region = (max(crop_region1[0], crop_region2[0]),
                                    max(crop_region1[1], crop_region2[1]),
                                    min(crop_region1[2], crop_region2[2]),
                                    min(crop_region1[3], crop_region2[3]))

                changed = False

                # subtract operation
                for i in range(intersect_region[0], intersect_region[2]):
                    for j in range(intersect_region[1], intersect_region[3]):
                        if cropped_mask1[j - crop_region1[1], i - crop_region1[0]] == 1 and \
                                cropped_mask2[j - crop_region2[1], i - crop_region2[0]] == 1:
                            # pixel overlaps with both masks, set it as 0
                            changed = True
                            cropped_mask1[j - crop_region1[1], i - crop_region1[0]] = 0
                        else:
                            # pixel does not overlap with both masks, don't care
                            pass

                if changed:
                    item = SEG(bseg.cropped_image, cropped_mask1, bseg.confidence, bseg.crop_region, bseg.bbox, bseg.label, None)
                    result.append(item)
                else:
                    result.append(base_segs)

        return ((base_segs[0], result),)

```
