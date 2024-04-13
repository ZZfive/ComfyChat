# EmptyLatentImagePresets
## Documentation
- Class name: `EmptyLatentImagePresets`
- Category: `KJNodes`
- Output node: `False`

The `EmptyLatentImagePresets` node provides a mechanism to generate latent images with predefined dimensions, offering an option to invert the width and height, and to specify the batch size for generation. This node simplifies the process of creating latent images by allowing users to select from a set of common dimensions, thereby streamlining the generation of images for various applications.
## Input types
### Required
- **`dimensions`**
    - Specifies the dimensions of the latent image to be generated. Users can select from a predefined list of common dimensions, with '512 x 512' being the default. This parameter determines the size of the generated latent image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`invert`**
    - A boolean flag that, when set to True, inverts the width and height of the specified dimensions. This allows for flexibility in the orientation of the generated latent image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`batch_size`**
    - Determines the number of latent images to generate in a single batch. This allows for efficient generation of multiple images at once, with a default value of 1 and a maximum of 4096.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`Latent`**
    - Comfy dtype: `LATENT`
    - The generated latent image.
    - Python dtype: `torch.Tensor`
- **`Width`**
    - Comfy dtype: `INT`
    - The width of the generated latent image.
    - Python dtype: `int`
- **`Height`**
    - Comfy dtype: `INT`
    - The height of the generated latent image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)



## Source code
```python
class EmptyLatentImagePresets:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
        "required": {
            "dimensions": (
            [   '512 x 512',
                '768 x 512',
                '960 x 512',
                '1024 x 512',
                '1536 x 640',
                '1344 x 768',
                '1216 x 832',
                '1152 x 896',
                '1024 x 1024',
            ],
            {
            "default": '512 x 512'
             }),
           
            "invert": ("BOOLEAN", {"default": False}),
            "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
            }),
        },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "KJNodes"

    def generate(self, dimensions, invert, batch_size):
        result = [x.strip() for x in dimensions.split('x')]
        
        if invert:
            width = int(result[1].split(' ')[0])
            height = int(result[0])
        else:
            width = int(result[0])
            height = int(result[1].split(' ')[0])
        latent = EmptyLatentImage().generate(width, height, batch_size)[0]

        return (latent, int(width), int(height),)

```
