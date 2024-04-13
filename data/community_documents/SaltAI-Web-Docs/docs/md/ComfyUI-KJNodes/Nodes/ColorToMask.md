# ColorToMask
## Documentation
- Class name: `ColorToMask`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to convert specific color values in images into masks, allowing for the isolation or removal of objects based on their color. It provides functionality to specify the color to be masked and adjust the sensitivity of the color matching process.
## Input types
### Required
- **`images`**
    - The images from which masks are to be created based on specified color values. This input is crucial for defining the source images for mask generation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`invert`**
    - Determines whether the generated mask should be inverted. This affects whether the specified color is masked out or preserved.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`red`**
    - The red component of the color to be masked, affecting the color matching process for mask generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`green`**
    - The green component of the color to be masked, affecting the color matching process for mask generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blue`**
    - The blue component of the color to be masked, affecting the color matching process for mask generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - The sensitivity of the color matching process, determining how closely colors must match the specified values to be included in the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask based on the specified color values and matching sensitivity.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorToMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "clip"
    CATEGORY = "KJNodes/masking"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "invert": ("BOOLEAN", {"default": False}),
                 "red": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "green": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "blue": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "threshold": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
        },
    } 

    def clip(self, images, red, green, blue, threshold, invert):
        color = np.array([red, green, blue])
        images = 255. * images.cpu().numpy()
        images = np.clip(images, 0, 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        images = [np.array(image) for image in images]

        black = [0, 0, 0]
        white = [255, 255, 255]
        if invert:
             black, white = white, black

        new_images = []
        for image in images:
            new_image = np.full_like(image, black)

            color_distances = np.linalg.norm(image - color, axis=-1)
            complement_indexes = color_distances <= threshold

            new_image[complement_indexes] = white

            new_images.append(new_image)

        new_images = np.array(new_images).astype(np.float32) / 255.0
        new_images = torch.from_numpy(new_images).permute(3, 0, 1, 2)
        return new_images

```
