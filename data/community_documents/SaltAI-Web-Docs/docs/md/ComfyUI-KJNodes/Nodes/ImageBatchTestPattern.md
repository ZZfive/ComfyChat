# ImageBatchTestPattern
## Documentation
- Class name: `ImageBatchTestPattern`
- Category: `KJNodes`
- Output node: `False`

The ImageBatchTestPattern node is designed to generate a series of test patterns for image batches. It allows for the creation of custom test images based on specified dimensions and starting points, facilitating the testing and debugging of image processing workflows.
## Input types
### Required
- **`batch_size`**
    - Specifies the number of test patterns to generate, affecting the overall size of the output batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_from`**
    - Determines the starting point for generating test patterns, enabling sequential or varied pattern generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Defines the width of each test pattern, setting the horizontal dimension of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of each test pattern, establishing the vertical dimension of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of test pattern images, each tailored to the specified dimensions and starting points, useful for testing and debugging purposes.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchTestPattern:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "batch_size": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
            "start_from": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),  
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generatetestpattern"
    CATEGORY = "KJNodes"

    def generatetestpattern(self, batch_size, start_from, width, height):
        out = []
        # Generate the sequential numbers for each image
        numbers = np.arange(batch_size) 

        # Create an image for each number
        for i, number in enumerate(numbers):
            # Create a black image with the number as a random color text
            image = Image.new("RGB", (width, height), color=0)
            draw = ImageDraw.Draw(image)

            # Draw a border around the image
            border_width = 10
            border_color = (255, 255, 255)  # white color
            border_box = [(border_width, border_width), (width - border_width, height - border_width)]
            draw.rectangle(border_box, fill=None, outline=border_color)
            
            font_size = 255  # Choose the desired font size
            font_path = "fonts\\TTNorms-Black.otf" #I don't know why relative path won't work otherwise...
            font_path = os.path.join(script_dir, font_path)
            
            # Generate a random color for the text
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            font = ImageFont.truetype(font_path, font_size)  # Replace "path_to_font_file.ttf" with the path to your font file
            text_width = font_size
            text_height = font_size
            text_x = (width - text_width / 2) // 2
            text_y = (height - text_height) // 2

            try:
                draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
            except:
                draw.text((text_x, text_y), text, font=font, fill=font_color)

            # Convert the image to a numpy array and normalize the pixel values
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            out.append(image)            

        return (torch.cat(out, dim=0),)

```
