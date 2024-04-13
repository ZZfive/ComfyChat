# Batch Shape (mtb)
## Documentation
- Class name: `Batch Shape (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The BatchShape node is designed to generate a batch of 2D shapes with optional shading, providing a versatile tool for creating diverse visual datasets. It abstracts the complexity of shape generation and shading, enabling users to easily produce batches of shapes with specified characteristics.
## Input types
### Required
- **`count`**
    - Specifies the number of shapes to generate in the batch, directly influencing the batch size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape`**
    - Determines the type of shapes to be generated in the batch, allowing for the creation of boxes, circles, or diamonds.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`image_width`**
    - Sets the width of the generated images, defining the canvas size for the shapes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_height`**
    - Sets the height of the generated images, defining the canvas size for the shapes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_size`**
    - Controls the size of the generated shapes, affecting their scale within the images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - Specifies the color of the shapes, enabling customization of their appearance.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`bg_color`**
    - Determines the background color of the images, setting the visual context for the shapes.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`shade_color`**
    - Defines the color used for shading the shapes, adding depth and visual interest.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`shadex`**
    - Adjusts the horizontal shading offset, influencing the direction and intensity of the shading effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`shadey`**
    - Adjusts the vertical shading offset, influencing the direction and intensity of the shading effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a list of images, each containing the generated shapes with the specified characteristics and optional shading.
    - Python dtype: `List[Image]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchShape:
    """Generates a batch of 2D shapes with optional shading (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1}),
                "shape": (
                    ["Box", "Circle", "Diamond"],
                    {"default": "Box"},
                ),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "shape_size": ("INT", {"default": 100}),
                "color": ("COLOR", {"default": "#ffffff"}),
                "bg_color": ("COLOR", {"default": "#000000"}),
                "shade_color": ("COLOR", {"default": "#000000"}),
                "shadex": ("FLOAT", {"default": 0.0}),
                "shadey": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_shapes"
    CATEGORY = "mtb/batch"

    def generate_shapes(
        self,
        count,
        shape,
        image_width,
        image_height,
        shape_size,
        color,
        bg_color,
        shade_color,
        shadex,
        shadey,
    ):
        print(f"COLOR: {color}")
        print(f"BG_COLOR: {bg_color}")
        print(f"SHADE_COLOR: {shade_color}")

        # Parse color input to BGR tuple for OpenCV
        color = hex_to_rgb(color)
        bg_color = hex_to_rgb(bg_color)
        shade_color = hex_to_rgb(shade_color)
        res = []
        for x in range(count):
            # Initialize an image canvas
            canvas = np.full(
                (image_height, image_width, 3), bg_color, dtype=np.uint8
            )
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Compute the center point of the shape
            center = (image_width // 2, image_height // 2)

            if shape == "Box":
                half_size = shape_size // 2
                top_left = (center[0] - half_size, center[1] - half_size)
                bottom_right = (center[0] + half_size, center[1] + half_size)
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            elif shape == "Circle":
                cv2.circle(mask, center, shape_size // 2, 255, -1)
            elif shape == "Diamond":
                pts = np.array(
                    [
                        [center[0], center[1] - shape_size // 2],
                        [center[0] + shape_size // 2, center[1]],
                        [center[0], center[1] + shape_size // 2],
                        [center[0] - shape_size // 2, center[1]],
                    ]
                )
                cv2.fillPoly(mask, [pts], 255)

            # Color the shape
            canvas[mask == 255] = color

            # Apply shading effects to a separate shading canvas
            shading = np.zeros_like(canvas, dtype=np.float32)
            shading[:, :, 0] = shadex * np.linspace(0, 1, image_width)
            shading[:, :, 1] = shadey * np.linspace(
                0, 1, image_height
            ).reshape(-1, 1)
            shading_canvas = cv2.addWeighted(
                canvas.astype(np.float32), 1, shading, 1, 0
            ).astype(np.uint8)

            # Apply shading only to the shape area using the mask
            canvas[mask == 255] = shading_canvas[mask == 255]
            res.append(canvas)

        return (pil2tensor(res),)

```
