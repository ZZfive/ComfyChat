# Smart Step (mtb)
## Documentation
- Class name: `Smart Step (mtb)`
- Category: `mtb/conditioning`
- Output node: `False`

Smart Step is designed to manage the progression of steps in the KAdvancedSampler by calculating start and end points based on given percentages. It provides a way to dynamically adjust the sampling process according to specified conditions.
## Input types
### Required
- **`step`**
    - The 'step' parameter specifies the current step in the sampling process. It is crucial for calculating the start and end points as percentages of this step value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_percent`**
    - The 'start_percent' parameter defines the percentage at which the sampling should start, relative to the total number of steps. It influences the calculation of the starting point in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_percent`**
    - The 'end_percent' parameter determines the percentage at which the sampling should end, relative to the total number of steps. It affects the calculation of the ending point in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`step`**
    - Comfy dtype: `INT`
    - Returns the original step value as an integer.
    - Python dtype: `int`
- **`start`**
    - Comfy dtype: `INT`
    - Returns the calculated start point as an integer, based on the 'start_percent' parameter.
    - Python dtype: `int`
- **`end`**
    - Comfy dtype: `INT`
    - Returns the calculated end point as an integer, based on the 'end_percent' parameter.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SmartStep:
    """Utils to control the steps start/stop of the KAdvancedSampler in percentage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "step": 1},
                ),
                "start_percent": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "end_percent": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("step", "start", "end")
    FUNCTION = "do_step"
    CATEGORY = "mtb/conditioning"

    def do_step(self, step, start_percent, end_percent):
        start = int(step * start_percent / 100)
        end = int(step * end_percent / 100)

        return (step, start, end)

```
