# Batch Float Assemble (mtb)
## Documentation
- Class name: `Batch Float Assemble (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The BatchFloatAssemble node is designed to aggregate multiple batches of floating-point numbers into a single, unified batch. This process can optionally reverse the order of the batches before combining them, allowing for flexible data manipulation and organization.
## Input types
### Required
- **`reverse`**
    - Determines whether the input batches should be reversed before being assembled into the final batch. This can affect the order in which data points are combined, potentially impacting downstream processing or analysis.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The assembled batch of floating-point numbers, which may have been reordered based on the 'reverse' parameter. This output is suitable for further numerical processing or analysis.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchFloatAssemble:
    """Assembles mutiple batches of floats into a single stream (batch)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse": ("BOOLEAN", {"default": False})}}

    FUNCTION = "assemble_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def assemble_floats(self, reverse, **kwargs):
        res = []
        if reverse:
            for x in reversed(kwargs.values()):
                res += x
        else:
            for x in kwargs.values():
                res += x

        return (res,)

```
