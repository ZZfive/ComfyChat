# 🔧 Draw Text
## Documentation
- Class name: `DrawText+`
- Category: `essentials`
- Output node: `False`

The DrawText+ node is designed to overlay text onto images, providing a wide range of customization options such as font selection, size, color, background color, shadow effects, and alignment. It enables users to enhance visual content by adding descriptive or artistic textual elements directly onto their images.
## Input types
### Required
- **`text`**
    - The text to be drawn on the image. This parameter allows users to specify the content of the textual overlay.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font`**
    - Specifies the font used for the text. This parameter enables customization of the text's appearance to match the image's aesthetic or the user's preference.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`size`**
    - Determines the size of the text. This parameter allows for the adjustment of the text's scale relative to the image, ensuring that it is appropriately proportioned.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - The color of the text. This parameter enables users to specify the text color, allowing for visual contrast or harmony with the image background.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`background_color`**
    - The background color of the text. This parameter allows for the addition of a background color to the text, enhancing readability or aesthetic appeal.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`shadow_distance`**
    - The distance of the shadow effect from the text. This parameter allows for the creation of a shadow effect, adding depth and emphasis to the text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_blur`**
    - The blur radius of the text shadow. This parameter controls the softness of the shadow effect, contributing to the text's visual impact.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_color`**
    - The color of the text shadow. This parameter enables customization of the shadow's color, further enhancing the text's visual appeal.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`alignment`**
    - Specifies the alignment of the text within the image. This parameter allows for the strategic placement of text to achieve the desired visual composition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`width`**
    - The width of the area where the text will be drawn. This parameter defines the horizontal bounds for text placement, ensuring it fits within the specified area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the area where the text will be drawn. This parameter defines the vertical bounds for text placement, ensuring it fits within the specified area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after text has been drawn onto it. This output is the visual content enhanced with the specified textual elements.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask indicating the areas where text has been added to the image. This output can be used for further processing or compositing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", { "multiline": True, "default": "Hello, World!" }),
                "font": ([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')], ),
                "size": ("INT", { "default": 56, "min": 1, "max": 9999, "step": 1 }),
                "color": ("STRING", { "multiline": False, "default": "#FFFFFF" }),
                "background_color": ("STRING", { "multiline": False, "default": "#00000000" }),
                "shadow_distance": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_blur": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_color": ("STRING", { "multiline": False, "default": "#000000" }),
                "alignment": (["left", "center", "right"],),
                "width": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
                "height": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height):
        font = ImageFont.truetype(os.path.join(FONTS_DIR, font), size)
        
        lines = text.split("\n")

        # Calculate the width and height of the text
        text_width = max(font.getbbox(line)[2] for line in lines)
        line_height = font.getmask(text).getbbox()[3] + font.getmetrics()[1]  # add descent to height
        text_height = line_height * len(lines)

        width = width if width > 0 else text_width
        height = height if height > 0 else text_height

        background_color = ImageColor.getrgb(background_color)
        image = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        image_shadow = None
        if shadow_distance > 0:
            image_shadow = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        for i, line in enumerate(lines):
            line_width = font.getbbox(line)[2]
            #text_height =font.getbbox(line)[3]
            if alignment == "left":
                x = 0
            elif alignment == "center":
                x = (width - line_width) / 2
            elif alignment == "right":
                x = width - line_width
            y = i * line_height

            draw = ImageDraw.Draw(image)
            draw.text((x, y), line, font=font, fill=color)
            
            if image_shadow is not None:
                draw = ImageDraw.Draw(image_shadow)
                draw.text((x + shadow_distance, y + shadow_distance), line, font=font, fill=shadow_color)

        if image_shadow is not None:
            image_shadow = image_shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
            image = Image.alpha_composite(image_shadow, image)

        image = pb(T.ToTensor()(image).unsqueeze(0))
        mask = image[:, :, :, 3] if image.shape[3] == 4 else torch.ones_like(image[:, :, :, 0])

        return (image[:, :, :, :3], mask,)

```
