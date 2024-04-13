# Adjust PE [Manual] 🎭🅐🅓
## Documentation
- Class name: `ADE_AdjustPEManual`
- Category: `Animate Diff 🎭🅐🅓/ad settings/pe adjust`
- Output node: `False`

The ADE_AdjustPEManual node allows for manual adjustment of positional encoding (PE) parameters within the AnimateDiff framework. It provides a flexible interface for users to specify various PE modification strategies, such as stretching, offsetting, or capping the length of PEs, to tailor the animation generation process to specific needs.
## Input types
### Required
- **`cap_initial_pe_length`**
    - Specifies the maximum length for the initial positional encoding. This parameter allows for capping the PE length at the beginning of the animation process, which can influence the animation's starting point or initial state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolate_pe_to_length`**
    - Determines the length to which the positional encoding should be interpolated. This adjustment can affect the smoothness and duration of the generated animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`initial_pe_idx_offset`**
    - Sets the starting index offset for the positional encoding. This parameter shifts the starting point of the positional encoding, potentially altering the animation's initial conditions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`final_pe_idx_offset`**
    - Defines the ending index offset for the positional encoding. It allows for adjustments to the ending conditions of the animation, affecting its final state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_adjustment`**
    - A boolean flag that, when set to true, enables the printing of adjustments made to the positional encoding. This feature aids in debugging and fine-tuning the animation parameters.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`prev_pe_adjust`**
    - An optional parameter that takes a previous positional encoding adjustment group. It allows for the accumulation and application of multiple PE adjustments over time.
    - Comfy dtype: `PE_ADJUST`
    - Python dtype: `AdjustPEGroup`
## Output types
- **`pe_adjust`**
    - Comfy dtype: `PE_ADJUST`
    - Returns an updated positional encoding adjustment group, incorporating the manual adjustments specified by the input parameters.
    - Python dtype: `AdjustPEGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ManualAdjustPENode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "print_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_pe_adjust": ("PE_ADJUST",),
            }
        }
    
    RETURN_TYPES = ("PE_ADJUST",)
    CATEGORY = "Animate Diff 🎭🅐🅓/ad settings/pe adjust"
    FUNCTION = "get_pe_adjust"

    def get_pe_adjust(self, cap_initial_pe_length: int, interpolate_pe_to_length: int, 
                      initial_pe_idx_offset: int, final_pe_idx_offset: int, print_adjustment: bool,
                      prev_pe_adjust: AdjustPEGroup=None):
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustPEGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length,
                          initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return (prev_pe_adjust,)

```
