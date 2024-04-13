# NormalizedAmplitudeToMask
## Documentation
- Class name: `NormalizedAmplitudeToMask`
- Category: `AudioScheduler/Amplitude`
- Output node: `False`

This node is designed to transform amplitude values into masks, applying normalization to ensure the amplitude values fall within a specific range. It focuses on manipulating amplitude data to create or adjust masks, which can be used in various image processing or machine learning tasks.
## Input types
### Required
- **`normalized_amp`**
    - The normalized amplitude values, expected to be within the range [0, 1]. These values are used to generate or modify masks based on the amplitude information.
    - Comfy dtype: `NORMALIZED_AMPLITUDE`
    - Python dtype: `numpy.ndarray`
- **`width`**
    - The desired width of the output mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The desired height of the output mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_offset`**
    - An integer value used to offset the amplitude values by rolling the array. This offset can be used to shift the mask generation process in a specific direction or manner.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_x`**
    - The x-coordinate location where the mask generation should start.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_y`**
    - The y-coordinate location where the mask generation should start.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`size`**
    - The size of the mask to be generated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape`**
    - The shape of the mask to be generated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`color`**
    - The color to be used in the mask generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask based on the input amplitude values and additional parameters.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NormalizedAmplitudeToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("NORMALIZED_AMPLITUDE",),
                    "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "frame_offset": ("INT", {"default": 0,"min": -255, "max": 255, "step": 1}),
                    "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "size": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                    "shape": (
                        [   
                            'none',
                            'circle',
                            'square',
                            'triangle',
                        ],
                        {
                        "default": 'none'
                        }),
                    "color": (
                        [   
                            'white',
                            'amplitude',
                        ],
                        {
                        "default": 'amplitude'
                        }),
                     },}

    CATEGORY = "AudioScheduler/Amplitude"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"

    def convert(self, normalized_amp, width, height, frame_offset, shape, location_x, location_y, size, color):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)

        # Offset the amplitude values by rolling the array
        normalized_amp = np.roll(normalized_amp, frame_offset)
        
        # Initialize an empty list to hold the image tensors
        out = []
        # Iterate over each amplitude value to create an image
        for amp in normalized_amp:
            # Scale the amplitude value to cover the full range of grayscale values
            if color == 'amplitude':
                grayscale_value = int(amp * 255)
            elif color == 'white':
                grayscale_value = 255
            # Convert the grayscale value to an RGB format
            gray_color = (grayscale_value, grayscale_value, grayscale_value)
            finalsize = size * amp
            
            if shape == 'none':
                shapeimage = Image.new("RGB", (width, height), gray_color)
            else:
                shapeimage = Image.new("RGB", (width, height), "black")

            draw = ImageDraw.Draw(shapeimage)
            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - finalsize, location_y - finalsize)
                right_down_point = (location_x + finalsize,location_y + finalsize)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=gray_color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=gray_color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - finalsize, location_y + finalsize) # bottom left
                right_down_point = (location_x + finalsize, location_y + finalsize) # bottom right
                top_point = (location_x, location_y) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=gray_color)
            
            shapeimage = pil2tensor(shapeimage)
            mask = shapeimage[:, :, :, 0]
            out.append(mask)
        
        return (torch.cat(out, dim=0),)

```
