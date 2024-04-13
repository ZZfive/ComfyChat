# Image Tile Offset (mtb)
## Documentation
- Class name: `Image Tile Offset (mtb)`
- Category: `mtb/generate`
- Output node: `False`

The Image Tile Offset node is designed to replicate a classic Photoshop technique for creating seamless textures by dividing an image into tiles and rearranging them. This process helps in identifying and correcting any visible seams in textures, making them ideal for repeated use in various digital environments.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image to be processed. It is crucial for determining the texture that will undergo the tiling and offsetting process to ensure seamless continuity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`tiles`**
    - The 'tiles' parameter specifies the number of divisions along both the height and width of the image, affecting the granularity of the tiling process. A higher number of tiles results in smaller, more numerous pieces, potentially offering a finer check for seamlessness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a modified version of the input image, rearranged into tiles to form a seamless texture. This technique is useful for texture artists and in applications requiring repeated background patterns.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageTileOffset:
    """Mimics an old photoshop technique to check for seamless textures"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiles": ("INT", {"default": 2}),
            }
        }

    CATEGORY = "mtb/generate"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "tile_image"

    def tile_image(self, image: torch.Tensor, tiles: int = 2):
        if tiles < 1:
            raise ValueError("The number of tiles must be at least 1.")

        batch_size, height, width, channels = image.shape
        tile_height = height // tiles
        tile_width = width // tiles

        output_image = torch.zeros_like(image)

        for i, j in itertools.product(range(tiles), range(tiles)):
            start_h = i * tile_height
            end_h = start_h + tile_height
            start_w = j * tile_width
            end_w = start_w + tile_width

            tile = image[:, start_h:end_h, start_w:end_w, :]

            output_start_h = (i + 1) % tiles * tile_height
            output_start_w = (j + 1) % tiles * tile_width
            output_end_h = output_start_h + tile_height
            output_end_w = output_start_w + tile_width

            output_image[
                :, output_start_h:output_end_h, output_start_w:output_end_w, :
            ] = tile

        return (output_image,)

```
