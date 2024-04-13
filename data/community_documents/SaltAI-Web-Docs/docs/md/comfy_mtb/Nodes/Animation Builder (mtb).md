# Animation Builder (mtb)
## Documentation
- Class name: `Animation Builder (mtb)`
- Category: `mtb/animation`
- Output node: `False`

The Animation Builder node provides a simple mathematical framework for generating and manipulating animation frames. It allows for the calculation of frame numbers, scaling of frames, and determining loop completion, facilitating the creation of smooth and customizable animations.
## Input types
### Required
- **`total_frames`**
    - Specifies the total number of frames in the animation, influencing the calculation of each frame's number and its scaled value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_float`**
    - A scaling factor applied to the frame number to adjust the animation's speed or progress.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`loop_count`**
    - Determines how many times the animation loop will repeat, affecting the overall length of the animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`raw_iteration`**
    - The current iteration number in the animation loop, used to calculate the current frame number.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`raw_loop`**
    - The current loop number, used to determine if the animation loop has ended.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`frame`**
    - Comfy dtype: `INT`
    - The current frame number in the animation sequence.
    - Python dtype: `int`
- **`0-1 (scaled)`**
    - Comfy dtype: `FLOAT`
    - The scaled value of the current frame, normalized between 0 and 1.
    - Python dtype: `float`
- **`count`**
    - Comfy dtype: `INT`
    - The current count of loops completed in the animation.
    - Python dtype: `int`
- **`loop_ended`**
    - Comfy dtype: `BOOLEAN`
    - A boolean indicating if the current animation loop has ended.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimationBuilder:
    """Simple maths for animation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 100, "min": 0}),
                # "fps": ("INT", {"default": 12, "min": 0}),
                "scale_float": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "loop_count": ("INT", {"default": 1, "min": 0}),
                "raw_iteration": ("INT", {"default": 0, "min": 0}),
                "raw_loop": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("frame", "0-1 (scaled)", "count", "loop_ended")
    CATEGORY = "mtb/animation"
    FUNCTION = "build_animation"

    def build_animation(
        self,
        total_frames=100,
        # fps=12,
        scale_float=1.0,
        loop_count=1,  # set in js
        raw_iteration=0,  # set in js
        raw_loop=0,  # set in js
    ):
        frame = raw_iteration % (total_frames)
        scaled = (frame / (total_frames - 1)) * scale_float
        # if frame == 0:
        #     log.debug("Reseting history")
        #     PromptServer.instance.prompt_queue.wipe_history()
        log.debug(f"frame: {frame}/{total_frames}  scaled: {scaled}")

        return (frame, scaled, raw_loop, (frame == (total_frames - 1)))

```
