# MorphologicOperation
## Documentation
- Class name: `MorphologicOperation`
- Category: `Bmad/CV/Morphology`
- Output node: `False`

This node provides a generic interface for performing various morphological operations on images, such as erosion, dilation, opening, closing, gradient, top hat, and black hat transformations. It is designed to work with binary (black and white) images and allows for the application of different kernel shapes and sizes to achieve the desired morphological effect.
## Input types
### Required
- **`src`**
    - The source image on which the morphological operations are to be performed. It serves as the primary input for the node, dictating the initial state of the image before any transformations. The quality and characteristics of the source image directly influence the outcome of the morphological operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`operation`**
    - Specifies the type of morphological operation to apply to the source image. This parameter determines how the image's structure will be modified, directly impacting the visual characteristics of the output image by altering its geometry and texture.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`kernel_type`**
    - Defines the shape of the structuring element used in the morphological operation. Different shapes can produce varying effects on the image, influencing the connectivity and the boundary smoothness of the structures within the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`kernel_size_x`**
    - The horizontal size of the kernel. It affects the extent of the operation's impact on the image, with larger sizes leading to more pronounced morphological transformations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`kernel_size_y`**
    - The vertical size of the kernel. Similar to kernel_size_x, it influences the scope of the morphological operation's effect on the image, where larger sizes result in more significant alterations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`iterations`**
    - The number of times the morphological operation is applied. Increasing the iterations can intensify the effect of the operation, leading to more dramatic changes in the image's structure.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed image after applying the specified morphological operation. It reflects the structural changes made to the original image, showcasing the effects of the chosen operation, kernel shape, and size.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MorphologicOperation:
    # I did not want to make this node, but alas, I found no suit w/ the top/black hat operation
    # so might as well make a generic node w/ all the operations
    # just return as BW and implement convert node

    operation_map = {
        "MORPH_ERODE": cv.MORPH_ERODE,
        "MORPH_DILATE": cv.MORPH_DILATE,
        "MORPH_OPEN": cv.MORPH_OPEN,
        "MORPH_CLOSE": cv.MORPH_CLOSE,
        "MORPH_GRADIENT": cv.MORPH_GRADIENT,
        "MORPH_TOPHAT": cv.MORPH_TOPHAT,
        "MORPH_BLACKHAT": cv.MORPH_BLACKHAT,
    }
    operations = list(operation_map.keys())

    kernel_types_map = {
        "MORPH_RECT": cv.MORPH_RECT,
        "MORPH_ELLIPSE": cv.MORPH_ELLIPSE,
        "MORPH_CROSS": cv.MORPH_CROSS,
    }
    kernel_types = list(kernel_types_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                "operation": (s.operations, s.operations),
                "kernel_type": (s.kernel_types, s.kernel_types),
                "kernel_size_x": ("INT", {"default": 4, "min": 2, "step": 2}),
                "kernel_size_y": ("INT", {"default": 4, "min": 2, "step": 2}),
                "iterations": ("INT", {"default": 1, "step": 1}),
            },
            # TODO maybe add optional input for custom kernel
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Bmad/CV/Morphology"

    def apply(self, src, operation, kernel_type, kernel_size_x, kernel_size_y, iterations):
        img = tensor2opencv(src, 1)
        kernel = cv.getStructuringElement(self.kernel_types_map[kernel_type], (kernel_size_x + 1, kernel_size_y + 1))
        for i in range(iterations):
            img = cv.morphologyEx(img, self.operation_map[operation], kernel)
        return (opencv2tensor(img),)

```
