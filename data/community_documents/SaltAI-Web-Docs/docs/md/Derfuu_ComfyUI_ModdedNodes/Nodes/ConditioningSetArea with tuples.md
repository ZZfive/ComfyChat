# ConditioningSetArea with tuples
## Documentation
- Class name: `ConditioningSetArea with tuples`
- Category: `Derfuu_Nodes/Tuples/Modded nodes/Conditioning`
- Output node: `False`

This node is designed to modify the conditioning of an input by setting specific areas within it. It takes tuples to define the size and offset of the area to be modified, along with a strength parameter to determine the intensity of the modification. The purpose is to allow for precise spatial adjustments within the conditioning, enhancing or altering specific regions as needed.
## Input types
### Required
- **`conditioning`**
    - The conditioning input represents the current state of conditioning that will be modified by this node. It is crucial as it serves as the base upon which modifications are applied.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`size_tuple`**
    - Defines the size of the area to be modified within the conditioning, using a tuple for width and height. This parameter is essential for specifying the spatial extent of the modification.
    - Comfy dtype: `TUPLE`
    - Python dtype: `Tuple[int, int]`
- **`offset_tuple`**
    - Specifies the starting point (x, y coordinates) of the area to be modified within the conditioning, allowing for precise location targeting.
    - Comfy dtype: `TUPLE`
    - Python dtype: `Tuple[int, int]`
- **`strength`**
    - Determines the intensity of the modification applied to the specified area within the conditioning. It affects how strongly the area's attributes are altered.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning, with specific areas adjusted according to the input parameters. It reflects the changes made to the original conditioning, showcasing the spatial modifications.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetArea:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "size_tuple": ("TUPLE",),
                "offset_tuple": ("TUPLE",),
                "strength": field.FLOAT,
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    CATEGORY = TREE_TUPLE_CONDITIONING

    def append(self, conditioning, size_tuple, offset_tuple, strength, min_sigma=0.0, max_sigma=99.0):

        width = int(size_tuple[0])
        height = int(size_tuple[1])

        x = int(offset_tuple[0])
        y = int(offset_tuple[1])

        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            # n[1]["area"].movedim(-1, 1)
            n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
            n[1]['strength'] = strength
            n[1]['min_sigma'] = min_sigma
            n[1]['max_sigma'] = max_sigma
            c.append(n)
        return (c,)

```
