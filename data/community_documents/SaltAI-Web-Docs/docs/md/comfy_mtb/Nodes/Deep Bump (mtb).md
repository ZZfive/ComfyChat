# Deep Bump (mtb)
## Documentation
- Class name: `Deep Bump (mtb)`
- Category: `mtb/textures`
- Output node: `False`

The Deep Bump node is designed for generating normal and height maps from single images. It supports various modes of operation, including converting color images to normal maps, generating curvature maps from normal maps, and converting normal maps to height maps. This node utilizes advanced image processing techniques to achieve detailed and accurate texture mappings, making it suitable for applications in texture generation and enhancement.
## Input types
### Required
- **`image`**
    - The input image for which the normal or height map is to be generated. This image serves as the base for all transformations and is crucial for the node's operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the operation mode of the node, which can be either 'Color to Normals', 'Normals to Curvature', or 'Normals to Height'. This determines the type of processing applied to the input image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`color_to_normals_overlap`**
    - Defines the overlap size when converting color images to normal maps. This parameter influences the smoothness and accuracy of the generated normal maps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`normals_to_curvature_blur_radius`**
    - Specifies the blur radius when generating curvature maps from normal maps. This affects the level of detail and smoothness in the curvature map.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`normals_to_height_seamless`**
    - A boolean flag indicating whether the conversion from normal maps to height maps should be seamless. This affects the continuity and visual quality of the generated height maps.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image, specifically either a normal map or a height map generated from the input image, depending on the selected mode of operation. This image can be used for further processing or visualization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DeepBump:
    """Normal & height maps generation from single pictures"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    ["Color to Normals", "Normals to Curvature", "Normals to Height"],
                ),
                "color_to_normals_overlap": (["SMALL", "MEDIUM", "LARGE"],),
                "normals_to_curvature_blur_radius": (
                    [
                        "SMALLEST",
                        "SMALLER",
                        "SMALL",
                        "MEDIUM",
                        "LARGE",
                        "LARGER",
                        "LARGEST",
                    ],
                ),
                "normals_to_height_seamless": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "mtb/textures"

    def apply(
        self,
        image,
        mode="Color to Normals",
        color_to_normals_overlap="SMALL",
        normals_to_curvature_blur_radius="SMALL",
        normals_to_height_seamless=True,
    ):
        images = tensor2pil(image)
        out_images = []

        for image in images:
            log.debug(f"Input image shape: {image}")

            in_img = np.transpose(image, (2, 0, 1)) / 255
            log.debug(f"transposed for deep image shape: {in_img.shape}")
            out_img = None

            # Apply processing
            if mode == "Color to Normals":
                out_img = color_to_normals(in_img, color_to_normals_overlap, None)
            if mode == "Normals to Curvature":
                out_img = normals_to_curvature(
                    in_img, normals_to_curvature_blur_radius, None
                )
            if mode == "Normals to Height":
                out_img = normals_to_height(in_img, normals_to_height_seamless, None)

            if out_img is not None:
                log.debug(f"Output image shape: {out_img.shape}")
                out_images.append(
                    torch.from_numpy(
                        np.transpose(out_img, (1, 2, 0)).astype(np.float32)
                    ).unsqueeze(0)
                )
            else:
                log.error("No out img... This should not happen")
        for outi in out_images:
            log.debug(f"Shape fed to utils: {outi.shape}")
        return (torch.cat(out_images, dim=0),)

```
