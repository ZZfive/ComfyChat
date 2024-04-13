# CreateTextMask
## Documentation
- Class name: `CreateTextMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

The CreateTextMask node is designed to generate masks based on textual content. It allows for the creation of masks that can be used to selectively apply effects or transformations to areas of an image that correspond to specific text.
## Input types
### Required
- **`invert`**
    - Determines whether the generated mask should be inverted. This affects the visual representation of the mask, with true values inverting the mask's color scheme.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frames`**
    - Specifies the number of frames for the generated mask. This is particularly useful for creating masks for animations or sequences of images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_x`**
    - The x-coordinate for the starting point of the text within the mask. This determines where the text begins horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_y`**
    - The y-coordinate for the starting point of the text within the mask. This determines where the text begins vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_size`**
    - The size of the font used for the text in the mask. This affects how large the text appears within the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_color`**
    - The color of the font used for the text. This determines the color of the text within the mask.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text`**
    - The text content to be used for generating the mask. This text will be visually represented in the mask.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font_path`**
    - The file path to the font used for the text. This allows for customization of the text's appearance based on the font style.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - The width of the mask to be generated. This determines the horizontal size of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the mask to be generated. This determines the vertical size of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_rotation`**
    - The starting rotation angle for the text within the mask. This allows for the text to be rotated to a specific angle at the beginning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_rotation`**
    - The ending rotation angle for the text within the mask. This allows for the text to be rotated to a different angle by the end.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated mask with textual content. This output is the primary result of the node, showcasing the mask created based on the specified text.
    - Python dtype: `Image`
- **`mask`**
    - Comfy dtype: `MASK`
    - An alternative output representing the mask in a format specifically suited for masking operations in image processing.
    - Python dtype: `Mask`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateTextMask:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                 "text_x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "text_y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "font_size": ("INT", {"default": 32,"min": 8, "max": 4096, "step": 1}),
                 "font_color": ("STRING", {"default": "white"}),
                 "text": ("STRING", {"default": "HELLO!"}),
                 "font_path": ("STRING", {"default": "fonts\\TTNorms-Black.otf"}),
                 "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "start_rotation": ("INT", {"default": 0,"min": 0, "max": 359, "step": 1}),
                 "end_rotation": ("INT", {"default": 0,"min": -359, "max": 359, "step": 1}),
        },
    } 

    def createtextmask(self, frames, width, height, invert, text_x, text_y, text, font_size, font_color, font_path, start_rotation, end_rotation):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        masks = []
        rotation = start_rotation
        if start_rotation != end_rotation:
            rotation_increment = (end_rotation - start_rotation) / (batch_size - 1)
        if font_path == "fonts\\TTNorms-Black.otf": #I don't know why relative path won't work otherwise...
            font_path = os.path.join(script_dir, font_path)
        # Generate the text
        for i in range(batch_size):
            image = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, font_size)
            text_width = font.getlength(text)
            text_height = font_size
            text_center_x = text_x + text_width / 2
            text_center_y = text_y + text_height / 2
            try:
                draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
            except:
                draw.text((text_x, text_y), text, font=font, fill=font_color)
            if start_rotation != end_rotation:
                image = image.rotate(rotation, center=(text_center_x, text_center_y))
                rotation += rotation_increment
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = image[:, :, :, 0] 
            masks.append(mask)
            out.append(image)
            
        if invert:
            return (1.0 - torch.cat(out, dim=0), 1.0 - torch.cat(masks, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)

```
