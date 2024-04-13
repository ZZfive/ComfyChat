# Batch Transform (mtb)
## Documentation
- Class name: `Batch Transform (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The Batch2dTransform node is designed to apply a series of 2D transformations to a batch of images. These transformations can include translation, rotation, zoom, and shear adjustments, allowing for dynamic and complex image manipulation.
## Input types
### Required
- **`image`**
    - The primary image tensor to which the batch of 2D transformations will be applied. This tensor serves as the foundation for the transformation process, dictating the visual content that will undergo modification.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`border_handling`**
    - Specifies how the borders of the image are handled during transformations, such as whether they are extended, reflected, or filled with a constant color. This affects the visual outcome of the transformation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`constant_color`**
    - Defines the color used to fill the borders of the image when 'constant' border handling is selected, influencing the aesthetic of the transformed image.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
### Optional
- **`x`**
    - Optional horizontal translation values for each image in the batch, enabling lateral movement.
    - Comfy dtype: `FLOATS`
    - Python dtype: `Optional[torch.Tensor]`
- **`y`**
    - Optional vertical translation values for each image in the batch, enabling vertical movement.
    - Comfy dtype: `FLOATS`
    - Python dtype: `Optional[torch.Tensor]`
- **`zoom`**
    - Optional zoom values for each image in the batch, allowing for image scaling.
    - Comfy dtype: `FLOATS`
    - Python dtype: `Optional[torch.Tensor]`
- **`angle`**
    - Optional rotation angles for each image in the batch, enabling rotational transformation.
    - Comfy dtype: `FLOATS`
    - Python dtype: `Optional[torch.Tensor]`
- **`shear`**
    - Optional shear values for each image in the batch, allowing for skewing transformations.
    - Comfy dtype: `FLOATS`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed batch of images, each having undergone the specified 2D transformations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Batch2dTransform:
    """Transform a batch of images using a batch of keyframes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {
                "x": ("FLOATS",),
                "y": ("FLOATS",),
                "zoom": ("FLOATS",),
                "angle": ("FLOATS",),
                "shear": ("FLOATS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_batch"
    CATEGORY = "mtb/batch"

    def get_num_elements(self, param) -> int:
        if isinstance(param, torch.Tensor):
            return torch.numel(param)
        elif isinstance(param, list):
            return len(param)

        return 0

    def transform_batch(
        self,
        image: torch.Tensor,
        border_handling,
        constant_color,
        x=None,
        y=None,
        zoom=None,
        angle=None,
        shear=None,
    ):
        if all(
            self.get_num_elements(param) <= 0
            for param in [x, y, zoom, angle, shear]
        ):
            raise ValueError(
                "At least one transform parameter must be provided"
            )

        keyframes = {"x": [], "y": [], "zoom": [], "angle": [], "shear": []}

        default_vals = {"x": 0, "y": 0, "zoom": 1.0, "angle": 0, "shear": 0}

        if self.get_num_elements(x) > 0:
            keyframes["x"] = x
        if self.get_num_elements(y) > 0:
            keyframes["y"] = y
        if self.get_num_elements(zoom) > 0:
            keyframes["zoom"] = zoom
        if self.get_num_elements(angle) > 0:
            keyframes["angle"] = angle
        if self.get_num_elements(shear) > 0:
            keyframes["shear"] = shear

        for name, values in keyframes.items():
            count = len(values)
            if count > 0 and count != image.shape[0]:
                raise ValueError(
                    f"Length of {name} values ({count}) must match number of images ({image.shape[0]})"
                )
            if count == 0:
                keyframes[name] = [default_vals[name]] * image.shape[0]

        transformer = TransformImage()
        res = [
            transformer.transform(
                image[i].unsqueeze(0),
                keyframes["x"][i],
                keyframes["y"][i],
                keyframes["zoom"][i],
                keyframes["angle"][i],
                keyframes["shear"][i],
                border_handling,
                constant_color,
            )[0]
            for i in range(image.shape[0])
        ]
        return (torch.cat(res, dim=0),)

```
