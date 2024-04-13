# ImageGridComposite3x3
## Documentation
- Class name: `ImageGridComposite3x3`
- Category: `KJNodes`
- Output node: `False`

The ImageGridComposite3x3 node is designed to create a composite image grid by arranging nine input images in a 3x3 matrix format. This node is particularly useful for visualizing multiple images in a structured layout, facilitating easier comparison or showcasing a collection of images in a unified manner.
## Input types
### Required
- **`image1`**
    - The first image to be placed in the top-left corner of the 3x3 grid. It serves as the starting point for the grid composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be placed in the top-center position of the 3x3 grid, contributing to the first row of the grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image3`**
    - The third image to be placed in the top-right corner of the 3x3 grid, completing the first row.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image4`**
    - The fourth image to be placed in the middle-left position of the 3x3 grid, starting the second row.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image5`**
    - The fifth image, positioned at the center of the 3x3 grid, serves as the focal point of the grid layout.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image6`**
    - The sixth image to be placed in the middle-right position of the 3x3 grid, completing the second row.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image7`**
    - The seventh image to be placed in the bottom-left corner of the 3x3 grid, starting the third row.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image8`**
    - The eighth image to be placed in the bottom-center position of the 3x3 grid, contributing to the final row.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image9`**
    - The ninth and final image to be placed in the bottom-right corner of the 3x3 grid, completing the grid composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The composite image created by arranging the nine input images in a 3x3 grid format.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageGridComposite3x3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
            "image5": ("IMAGE",),
            "image6": ("IMAGE",),
            "image7": ("IMAGE",),
            "image8": ("IMAGE",),
            "image9": ("IMAGE",),     
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "KJNodes"

    def compositegrid(self, image1, image2, image3, image4, image5, image6, image7, image8, image9):
        top_row = torch.cat((image1, image2, image3), dim=2)
        mid_row = torch.cat((image4, image5, image6), dim=2)
        bottom_row = torch.cat((image7, image8, image9), dim=2)
        grid = torch.cat((top_row, mid_row, bottom_row), dim=1)
        return (grid,)

```
