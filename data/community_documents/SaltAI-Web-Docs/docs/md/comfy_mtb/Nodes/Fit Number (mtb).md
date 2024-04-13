# Fit Number (mtb)
## Documentation
- Class name: `Fit Number (mtb)`
- Category: `mtb/math`
- Output node: `False`

The Fit Number node is designed to adapt numerical values to a specified range or format, ensuring they fit within certain constraints or match a desired pattern. This node is essential for preprocessing numerical data, making it suitable for further analysis or modeling by adjusting its scale or format.
## Input types
### Required
- **`value`**
    - Specifies the numerical value to be adapted. It's crucial for determining the starting point of the adaptation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clamp`**
    - Determines whether the output value should be clamped within the target range, ensuring it does not exceed the specified limits.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`source_min`**
    - Defines the minimum value of the source range, serving as a reference point for scaling the input value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`source_max`**
    - Sets the maximum value of the source range, used alongside source_min to scale the input value appropriately.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`target_min`**
    - Specifies the minimum value of the target range, indicating the lower bound to which the input value is scaled.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`target_max`**
    - Defines the maximum value of the target range, establishing the upper limit for the scaled input value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`easing`**
    - Selects the easing function to apply during the adaptation process, affecting how the input value transitions between the source and target ranges.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The adapted numerical value after being processed through the specified source and target ranges, including any applied easing functions.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FitNumber:
    """Fit the input float using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    ],
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "mtb/math"
    DESCRIPTION = "Fit the input float using a source and target range"

    def set_range(
        self,
        value: float,
        clamp: bool,
        source_min: float,
        source_max: float,
        target_min: float,
        target_max: float,
        easing: str,
    ):
        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)

        eased_value = apply_easing(normalized_value, easing)

        # - Convert the eased value to the target range
        res = target_min + (target_max - target_min) * eased_value

        return (res,)

```
