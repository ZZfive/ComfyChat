# Picker (SEGS)
## Documentation
- Class name: `ImpactSEGSPicker`
- Category: `ImpactPack/Util`
- Output node: `True`

The ImpactSEGSPicker node is designed to select specific segments from a collection based on user-defined criteria, and optionally adjust their appearance using fallback images or masks. It facilitates the customization and refinement of segment collections for further processing or visualization.
## Input types
### Required
- **`picks`**
    - Specifies the indices of the segments to be selected from the collection. It directly influences which segments are retained for further operations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`segs`**
    - The collection of segments from which selections are made. It serves as the primary input for the node's selection process.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
### Optional
- **`fallback_image_opt`**
    - An optional image used to modify the appearance of segments when their original images are not available.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[PIL.Image]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - Returns a tuple containing the original collection identifier and the new collection of selected and optionally modified segments.
    - Python dtype: `Tuple[Any, List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SEGSPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "picks": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                    "segs": ("SEGS",),
                    },
                "optional": {
                     "fallback_image_opt": ("IMAGE", ),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("SEGS", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, picks, segs, fallback_image_opt=None, unique_id=None):
        if fallback_image_opt is not None:
            segs = core.segs_scale_match(segs, fallback_image_opt.shape)

        # generate candidates image
        cands = []
        for seg in segs[1]:
            if seg.cropped_image is not None:
                cropped_image = seg.cropped_image
            elif fallback_image_opt is not None:
                # take from original image
                cropped_image = crop_image(fallback_image_opt, seg.crop_region)
            else:
                cropped_image = empty_pil_tensor()

            mask_array = seg.cropped_mask
            mask_array[mask_array < 0.3] = 0.3
            mask_array = mask_array[None, ..., None]
            cropped_image = cropped_image * mask_array

            cands.append(cropped_image)

        impact.impact_server.segs_picker_map[unique_id] = cands

        # pass only selected
        pick_ids = set()

        for pick in picks.split(","):
            try:
                pick_ids.add(int(pick)-1)
            except Exception:
                pass

        new_segs = []
        for i in pick_ids:
            if 0 <= i < len(segs[1]):
                new_segs.append(segs[1][i])

        return ((segs[0], new_segs),)

```
