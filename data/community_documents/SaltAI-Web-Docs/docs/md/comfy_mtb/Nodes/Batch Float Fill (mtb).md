# Batch Float Fill (mtb)
## Documentation
- Class name: `Batch Float Fill (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The BatchFloatFill node is designed to extend a batch of floating-point numbers to a specified length by filling it with a predefined value, either at the beginning or the end of the batch. This functionality is crucial for ensuring uniform batch sizes or for padding purposes in various data processing workflows.
## Input types
### Required
- **`floats`**
    - The initial batch of floating-point numbers to be filled. This parameter is the starting point for the operation, determining the base that will be extended to reach the target length.
    - Comfy dtype: `FLOATS`
    - Python dtype: `List[float]`
- **`direction`**
    - Specifies whether the filling should occur at the beginning ('head') or the end ('tail') of the batch. This affects the order in which the batch is extended.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`value`**
    - The floating-point number to be used for filling the batch. This value is replicated to reach the desired count.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`count`**
    - The target length for the batch after filling. This determines how many times the value will be added to reach the specified size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The resulting batch of floating-point numbers after filling to the target length.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchFloatFill:
    """Fills a batch float with a single value until it reaches the target length"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS",),
                "direction": (["head", "tail"], {"default": "tail"}),
                "value": ("FLOAT", {"default": 0.0}),
                "count": ("INT", {"default": 1}),
            }
        }

    FUNCTION = "fill_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def fill_floats(self, floats, direction, value, count):
        size = len(floats)
        if size > count:
            raise ValueError(
                f"Size ({size}) is less then target count ({count})"
            )

        rem = count - size
        if direction == "tail":
            floats = floats + [value] * rem
        else:
            floats = [value] * rem + floats
        return (floats,)

```
