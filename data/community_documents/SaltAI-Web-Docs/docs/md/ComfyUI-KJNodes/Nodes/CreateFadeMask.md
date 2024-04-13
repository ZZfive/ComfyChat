# CreateFadeMask
## Documentation
- Class name: `CreateFadeMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

The CreateFadeMask node is designed to generate a series of masks that transition in opacity, allowing for the creation of fade effects in animations or image sequences. It supports customization of the fade pattern through various parameters, including the ability to invert the fade, specify the number of frames, and define the interpolation method.
## Input types
### Required
- **`invert`**
    - Determines whether the fade effect should be inverted, affecting the direction of the fade transition in the generated mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frames`**
    - Specifies the number of frames for the fade mask sequence, controlling the length of the fade effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Sets the width of the generated fade mask, defining the horizontal dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated fade mask, defining the vertical dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - Chooses the interpolation method for the fade effect, affecting how the opacity transitions between frames.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_level`**
    - Defines the starting opacity level for the fade effect, determining the initial visibility of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`midpoint_level`**
    - Sets the opacity level at the midpoint of the fade effect, allowing for control over the fade's intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_level`**
    - Determines the ending opacity level for the fade effect, defining the final visibility of the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`midpoint_frame`**
    - Specifies the frame at which the midpoint opacity level is reached, controlling the timing of the fade effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated fade mask, which transitions in opacity according to the specified parameters.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateFadeMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "createfademask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 2,"min": 2, "max": 255, "step": 1}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                 "start_level": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "midpoint_level": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "end_level": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "midpoint_frame": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
        },
    } 
    
    def createfademask(self, frames, width, height, invert, interpolation, start_level, midpoint_level, end_level, midpoint_frame):
        def ease_in(t):
            return t * t

        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        if midpoint_frame == 0:
            midpoint_frame = batch_size // 2

        for i in range(batch_size):
            if i <= midpoint_frame:
                t = i / midpoint_frame
                if interpolation == "ease_in":
                    t = ease_in(t)
                elif interpolation == "ease_out":
                    t = ease_out(t)
                elif interpolation == "ease_in_out":
                    t = ease_in_out(t)
                color = start_level - t * (start_level - midpoint_level)
            else:
                t = (i - midpoint_frame) / (batch_size - midpoint_frame)
                if interpolation == "ease_in":
                    t = ease_in(t)
                elif interpolation == "ease_out":
                    t = ease_out(t)
                elif interpolation == "ease_in_out":
                    t = ease_in_out(t)
                color = midpoint_level - t * (midpoint_level - end_level)

            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

```
