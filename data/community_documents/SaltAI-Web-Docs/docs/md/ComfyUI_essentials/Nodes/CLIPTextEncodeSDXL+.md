# 🔧 SDXLCLIPTextEncode
## Documentation
- Class name: `CLIPTextEncodeSDXL+`
- Category: `essentials`
- Output node: `False`

This node is designed to encode text inputs using a CLIP model specifically tailored for the SDXL architecture. It focuses on generating conditioning vectors and associated metadata from textual descriptions, facilitating advanced image generation tasks by providing detailed control over the aesthetic and dimensional aspects of the output.
## Input types
### Required
- **`width`**
    - Specifies the desired width of the output image, playing a crucial role in determining the dimensions of the generated visual content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output image, directly affecting the vertical dimension of the generated visual content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text`**
    - The textual description that will be encoded into a conditioning vector, serving as the basis for generating images that match the described content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - Specifies the CLIP model to be used for text tokenization and encoding, essential for transforming text descriptions into conditioning vectors.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - A conditioning vector along with associated metadata, including image dimensions, tailored for guiding the image generation process.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, float]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPTextEncodeSDXLSimplified:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, clip, width, height, text):
        crop_w = 0
        crop_h = 0
        width = width*4
        height = height*4
        target_width = width
        target_height = height
        text_g = text_l = text

        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )

```
