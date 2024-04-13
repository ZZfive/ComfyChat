# GLIGENTextBoxApply
## Documentation
- Class name: `GLIGENTextBoxApply`
- Category: `conditioning/gligen`
- Output node: `False`

The GLIGENTextBoxApply node is designed to integrate text-based conditioning into a generative model's input, specifically by applying text box parameters and encoding them using a CLIP model. This process enriches the conditioning with spatial and textual information, facilitating more precise and context-aware generation.
## Input types
### Required
- **`conditioning_to`**
    - Specifies the initial conditioning input to which the text box parameters and encoded text information will be appended. It plays a crucial role in determining the final output by integrating new conditioning data.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`clip`**
    - The CLIP model used for encoding the provided text into a format that can be utilized by the generative model. It's essential for converting textual information into a compatible conditioning format.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
- **`gligen_textbox_model`**
    - Represents the specific GLIGEN model configuration to be used for generating the text box. It's crucial for ensuring that the text box is generated according to the desired specifications.
    - Comfy dtype: `GLIGEN`
    - Python dtype: `GLIGEN`
- **`text`**
    - The text content to be encoded and integrated into the conditioning. It provides the semantic information that guides the generative model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - The width of the text box in pixels. It defines the spatial dimension of the text box within the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the text box in pixels. Similar to width, it defines the spatial dimension of the text box within the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`x`**
    - The x-coordinate of the top-left corner of the text box within the generated image. It specifies the text box's position horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The y-coordinate of the top-left corner of the text box within the generated image. It specifies the text box's position vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The enriched conditioning output, which includes the original conditioning data along with the newly appended text box parameters and encoded text information. It's used to guide the generative model in producing context-aware outputs.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [ToDetailerPipe](../../ComfyUI-Impact-Pack/Nodes/ToDetailerPipe.md)



## Source code
```python
class GLIGENTextBoxApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ),
                              "clip": ("CLIP", ),
                              "gligen_textbox_model": ("GLIGEN", ),
                              "text": ("STRING", {"multiline": True}),
                              "width": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                              "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                              "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "conditioning/gligen"

    def append(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y):
        c = []
        cond, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled="unprojected")
        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            position_params = [(cond_pooled, height // 8, width // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", gligen_textbox_model, prev + position_params)
            c.append(n)
        return (c, )

```
