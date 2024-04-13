# Default Iteration Options 🎭🅐🅓
## Documentation
- Class name: `ADE_IterationOptsDefault`
- Category: `Animate Diff 🎭🅐🅓/iteration opts`
- Output node: `False`

This node is designed to generate and manage iteration options for the AnimateDiff process, allowing for customization of iteration parameters such as the number of iterations and various offsets. It plays a crucial role in defining how the animation diffing process iterates over frames or elements, providing flexibility in the animation generation workflow.
## Input types
### Required
- **`iterations`**
    - Specifies the number of iterations to perform, directly influencing the depth or extent of the animation diffing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`iter_batch_offset`**
    - Determines the offset for batch processing within iterations, enabling fine-tuned control over the iteration sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`iter_seed_offset`**
    - Sets the seed offset for iterations, affecting the randomness or variability in the animation diffing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`iteration_opts`**
    - Comfy dtype: `ITERATION_OPTS`
    - Outputs the configured iteration options, encapsulating all specified parameters for use in the animation diffing process.
    - Python dtype: `IterationOptions`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IterationOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "iterations": ("INT", {"default": 1, "min": 1}),
            },
            "optional": {
                "iter_batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "iter_seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("ITERATION_OPTS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/iteration opts"
    FUNCTION = "create_iter_opts"

    def create_iter_opts(self, iterations: int, iter_batch_offset: int=0, iter_seed_offset: int=0):
        iter_opts = IterationOptions(iterations=iterations, iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return (iter_opts,)

```
