# CreateShapeMask
## Documentation
- Class name: `CreateShapeMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

The CreateShapeMask node is designed to generate masks based on geometric shapes such as circles, squares, and triangles. It allows for the creation of shape-based masks by specifying the shape, size, and location, enabling the customization of masks for various image processing and computer vision tasks.
## Input types
### Required
- **`shape`**
    - Specifies the geometric shape of the mask to be created. It determines the overall form and structure of the resulting mask, playing a crucial role in the mask's appearance and utility.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - Defines the number of frames for the mask, allowing for the creation of multiple masks in a sequence. This can be useful for generating masks for animations or video processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_x`**
    - Determines the x-coordinate of the shape's location within the mask. It affects the horizontal positioning of the shape, influencing the mask's spatial arrangement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_y`**
    - Determines the y-coordinate of the shape's location within the mask. It affects the vertical positioning of the shape, influencing the mask's spatial arrangement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow`**
    - Specifies the amount by which the shape should grow or shrink. This parameter allows for dynamic adjustment of the shape's size, enabling more flexible mask creation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_width`**
    - Sets the width of the frame for the mask. This determines the horizontal dimension of the mask, affecting its size and aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - Sets the height of the frame for the mask. This determines the vertical dimension of the mask, affecting its size and aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_width`**
    - Specifies the width of the shape within the mask. This parameter allows for precise control over the size of the shape, affecting its visibility and impact on the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_height`**
    - Specifies the height of the shape within the mask. This parameter allows for precise control over the size of the shape, affecting its visibility and impact on the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask generated based on the specified geometric shape. It serves as a customizable shape-based mask for various applications.
    - Python dtype: `torch.Tensor`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - The output is an inverted version of the mask generated based on the specified geometric shape. This inversion can be useful for creating complementary masks for certain applications.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [MaskToImage](../../Comfy/Nodes/MaskToImage.md)



## Source code
```python
class CreateShapeMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
    } 

    def createshapemask(self, frames, frame_width, frame_height, location_x, location_y, shape_width, shape_height, grow, shape):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        color = "white"
        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i*grow)
            current_height = max(0, shape_height + i*grow)

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=color)

            image = pil2tensor(image)
            mask = image[:, :, :, 0]
            out.append(mask)
        return (torch.cat(out, dim=0), 1.0 - torch.cat(out, dim=0),)

```
