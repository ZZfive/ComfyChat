# ImageGridComposite2x2
## Documentation
- Class name: `ImageGridComposite2x2`
- Category: `KJNodes`
- Output node: `False`

The ImageGridComposite2x2 node is designed to create a 2x2 grid composite image from four input images. It arranges the images in a grid layout, concatenating them vertically and horizontally to form a single composite image.
## Input types
### Required
- **`image1`**
    - The first image to be placed in the top-left corner of the 2x2 grid. It contributes to the overall composition by starting the visual sequence.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be placed in the top-right corner of the 2x2 grid. It complements the first image by continuing the visual sequence.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image3`**
    - The third image to be placed in the bottom-left corner of the 2x2 grid. It adds to the composite by extending the visual narrative below the first image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image4`**
    - The fourth image to be placed in the bottom-right corner of the 2x2 grid. It completes the visual sequence, forming a cohesive composite image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting composite image formed by arranging the four input images into a 2x2 grid layout.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageGridComposite2x2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),   
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compositegrid"
    CATEGORY = "KJNodes"

    def compositegrid(self, image1, image2, image3, image4):
        top_row = torch.cat((image1, image2), dim=2)
        bottom_row = torch.cat((image3, image4), dim=2)
        grid = torch.cat((top_row, bottom_row), dim=1)
        return (grid,)

```
