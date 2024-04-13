# 🔧 Transition Mask
## Documentation
- Class name: `TransitionMask+`
- Category: `essentials`
- Output node: `False`

This node is designed to perform operations on masks, allowing for the combination of two masks through various mathematical operations. It enables the manipulation of mask data to achieve desired effects, such as blending, masking, or creating new mask patterns.
## Input types
### Required
- **`width`**
    - Specifies the width of the mask to be generated or modified.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the mask to be generated or modified.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frames`**
    - Determines the number of frames for the transition effect between masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_frame`**
    - Specifies the starting frame number for the transition effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - Specifies the ending frame number for the transition effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`transition_type`**
    - Defines the type of transition effect to be applied between the masks.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`timing_function`**
    - Specifies the timing function to control the pacing of the transition effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The result of combining the two input masks according to the specified operation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TransitionMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "timing_function": (["linear", "in", "out", "in-out"],)
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, width, height, frames, start_frame, end_frame, transition_type, timing_function):
        if timing_function == 'in':
            tf = [0.0, 0.0, 0.5, 1.0]
        elif timing_function == 'out':
            tf = [0.0, 0.5, 1.0, 1.0]
        elif timing_function == 'in-out':
            tf = [0, 1, 0, 1]
        #elif timing_function == 'back':
        #    tf = [0, 1.334, 1.334, 0]
        else:
            tf = [0, 0, 1, 1]

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = i/(transition-1)

            if timing_function != 'linear':
                progress = cubic_bezier(progress, tf)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)
        
        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)
           
        return (out, )

```
