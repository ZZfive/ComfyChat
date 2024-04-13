# CLIPTextEncodeSDXLRefiner
## Documentation
- Class name: `CLIPTextEncodeSDXLRefiner`
- Category: `advanced/conditioning`
- Output node: `False`

This node specializes in refining the encoding of text inputs using CLIP models, enhancing the conditioning for generative tasks by incorporating aesthetic scores and dimensions.
## Input types
### Required
- **`ascore`**
    - The aesthetic score parameter influences the conditioning output by providing a measure of aesthetic quality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`width`**
    - Specifies the width of the output conditioning, affecting the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output conditioning, influencing the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text`**
    - The text input to be encoded, serving as the primary content descriptor for conditioning.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A CLIP model instance used for text tokenization and encoding, central to generating the conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The refined conditioning output, enriched with aesthetic scores and dimensions for enhanced content generation.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - UltimateSDUpscale



## Source code
```python
class CLIPTextEncodeSDXLRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "aesthetic_score": ascore, "width": width,"height": height}]], )

```
