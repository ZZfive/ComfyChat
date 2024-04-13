# AddLabel
## Documentation
- Class name: `AddLabel`
- Category: `KJNodes`
- Output node: `False`

The AddLabel node is designed to overlay text labels onto images, allowing for customization of text position, size, color, and the label's background color. It supports specifying the direction in which the label should be added relative to the image, enabling versatile text annotation capabilities.
## Input types
### Required
- **`image`**
    - The input image or batch of images to which the label will be added. This parameter is crucial as it defines the base upon which the text label will be overlaid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`text_x`**
    - The x-coordinate for the starting point of the text on the image. It determines where horizontally the text will begin.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_y`**
    - The y-coordinate for the starting point of the text on the image. It determines where vertically the text will begin.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the text label area in pixels. It affects how much vertical space the label will occupy on the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_size`**
    - The size of the font used for the text label. This parameter influences the readability and visual impact of the text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_color`**
    - The color of the text in the label. It determines how the text visually contrasts with the label's background.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`label_color`**
    - The background color of the text label. This parameter affects the visual distinction between the label and the image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font`**
    - The font used for the text label. This parameter allows for customization of the text's appearance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text`**
    - The text content to be added as a label on the image. This parameter is essential for defining what the label will say.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`direction`**
    - The direction in which the label should be added relative to the image, either 'up' or 'down'. This affects the label's placement and can influence the overall layout of the annotated image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images with the text labels added in the specified direction. It enables the integration of textual information directly onto visual content.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AddLabel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image":("IMAGE",),  
            "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 48, "min": 0, "max": 4096, "step": 1}),
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
            "font_color": ("STRING", {"default": "white"}),
            "label_color": ("STRING", {"default": "black"}),
            "font": ("STRING", {"default": "TTNorms-Black.otf"}),
            "text": ("STRING", {"default": "Text"}),
            "direction": (
            [   'up',
                'down',
            ],
            {
            "default": 'up'
            
             }),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "addlabel"

    CATEGORY = "KJNodes"
        
    def addlabel(self, image, text_x, text_y, text, height, font_size, font_color, label_color, font, direction):
        batch_size = image.shape[0]
        width = image.shape[2]
        
        if font == "TTNorms-Black.otf":
            font_path = os.path.join(script_dir, "fonts", "TTNorms-Black.otf")
        else:
            font_path = font
        label_image = Image.new("RGB", (width, height), label_color)
        draw = ImageDraw.Draw(label_image)
        font = ImageFont.truetype(font_path, font_size)
        try:
            draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
        except:
            draw.text((text_x, text_y), text, font=font, fill=font_color)

        label_image = np.array(label_image).astype(np.float32) / 255.0
        label_image = torch.from_numpy(label_image)[None, :, :, :]
        # Duplicate the label image for the entire batch
        label_batch = label_image.repeat(batch_size, 1, 1, 1)

        if direction == 'down':
            combined_images = torch.cat((image, label_batch), dim=1)
        elif direction == 'up':
            combined_images = torch.cat((label_batch, image), dim=1)
        
        return (combined_images,)

```
