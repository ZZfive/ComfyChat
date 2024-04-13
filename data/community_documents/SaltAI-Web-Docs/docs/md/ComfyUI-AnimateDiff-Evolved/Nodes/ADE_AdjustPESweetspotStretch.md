# Adjust PE [Sweetspot Stretch] 🎭🅐🅓
## Documentation
- Class name: `ADE_AdjustPESweetspotStretch`
- Category: `Animate Diff 🎭🅐🅓/ad settings/pe adjust`
- Output node: `False`

This node is designed to adjust the positional encoding (PE) sweetspot in the context of animating differences, allowing for the modification of the initial sweetspot and the introduction of a new sweetspot value. It facilitates the fine-tuning of animation parameters to achieve desired effects.
## Input types
### Required
- **`sweetspot`**
    - Specifies the original sweetspot value for positional encoding, serving as a baseline for adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`new_sweetspot`**
    - Defines the new sweetspot value to which the positional encoding should be adjusted, enabling the customization of animation dynamics.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_adjustment`**
    - A boolean flag that, when set to True, enables the printing of adjustment details for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`prev_pe_adjust`**
    - An optional parameter that allows for the incorporation of previous positional encoding adjustments, facilitating cumulative adjustments.
    - Comfy dtype: `PE_ADJUST`
    - Python dtype: `AdjustPEGroup`
## Output types
- **`pe_adjust`**
    - Comfy dtype: `PE_ADJUST`
    - Returns the updated positional encoding adjustment group, encapsulating the applied sweetspot adjustments.
    - Python dtype: `AdjustPEGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SweetspotStretchPENode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sweetspot": ("INT", {"default": 16, "min": 0, "max": BIGMAX},),
                "new_sweetspot": ("INT", {"default": 16, "min": 0, "max": BIGMAX},),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_pe_adjust": ("PE_ADJUST",),
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, sweetspot: int, new_sweetspot: int, print_adjustment: bool, prev_pe_adjust: AdjustPEGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustPEGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=sweetspot, interpolate_pe_to_length=new_sweetspot,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)

```
