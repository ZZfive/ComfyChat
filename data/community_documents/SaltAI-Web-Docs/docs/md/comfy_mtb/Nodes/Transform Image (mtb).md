# Transform Image (mtb)
## Documentation
- Class name: `Transform Image (mtb)`
- Category: `mtb/transform`
- Output node: `False`

This node is designed to apply a series of geometric transformations to an image, including translation, rotation, scaling, and shearing. It allows for flexible image manipulation with options for border handling and color filling, making it suitable for a wide range of image processing tasks.
## Input types
### Required
- **`image`**
    - The input image to be transformed, provided as a tensor. This image undergoes various geometric transformations based on the other parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`x`**
    - The horizontal translation of the image, affecting its position along the x-axis.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`y`**
    - The vertical translation of the image, affecting its position along the y-axis.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom`**
    - The scaling factor for the image, allowing for zooming in or out.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle`**
    - The rotation angle of the image in degrees, enabling its rotation around the center.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`shear`**
    - The shear angle of the image, allowing for a shearing transformation to be applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`border_handling`**
    - Specifies how the borders of the image are handled during transformations, such as 'edge' or 'constant'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`constant_color`**
    - The color used to fill the constant areas when 'constant' border handling is selected. It's specified as a hex color code.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed image, returned as a tensor after applying the specified geometric transformations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TransformImage:
    """Save torch tensors (image, mask or latent) to disk, useful to debug things outside comfy


    it return a tensor representing the transformed images with the same shape as the input tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "y": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "shear": (
                    "FLOAT",
                    {"default": 0, "step": 1, "min": -4096, "max": 4096},
                ),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mtb/transform"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling="edge",
        constant_color=None,
    ):
        x = int(x)
        y = int(y)
        angle = int(angle)

        log.debug(f"Zoom: {zoom} | x: {x}, y: {y}, angle: {angle}, shear: {shear}")

        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        log.debug(f"New height: {new_height}, New width: {new_width}")

        # - Calculate diagonal of the original image
        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [max(0, pw + x), max(0, ph + y), max(0, pw - x), max(0, ph - y)]

        constant_color = hex_to_rgb(constant_color)
        log.debug(f"Fill Tuple: {constant_color}")

        for img in tensor2pil(image):
            img = TF.pad(
                img,  # transformed_frame,
                padding=padding,
                padding_mode=border_handling,
                fill=constant_color or 0,
            )

            img = cast(
                Image.Image,
                TF.affine(img, angle=angle, scale=zoom, translate=[x, y], shear=shear),
            )

            left = abs(padding[0])
            upper = abs(padding[1])
            right = img.width - abs(padding[2])
            bottom = img.height - abs(padding[3])

            # log.debug("crop is [:,top:bottom, left:right] for tensors")
            log.debug("crop is [left, top, right, bottom] for PIL")
            log.debug(f"crop is {left}, {upper}, {right}, {bottom}")
            img = img.crop((left, upper, right, bottom))

            transformed_images.append(img)

        return (pil2tensor(transformed_images),)

```
