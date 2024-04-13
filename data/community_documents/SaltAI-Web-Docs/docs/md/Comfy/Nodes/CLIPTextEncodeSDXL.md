# CLIPTextEncodeSDXL
## Documentation
- Class name: `CLIPTextEncodeSDXL`
- Category: `advanced/conditioning`
- Output node: `False`

This node is designed to encode text inputs using the CLIP model specifically tailored for the SDXL architecture. It focuses on converting textual descriptions into a format that can be effectively utilized for generating or manipulating images, leveraging the capabilities of the CLIP model to understand and process text in the context of visual content.
## Input types
### Required
- **`width`**
    - Specifies the width of the image in pixels. It determines the dimensions of the output image generated or manipulated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the image in pixels. It determines the dimensions of the output image generated or manipulated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Defines the width of the crop area in pixels. This parameter is used to crop the image to a specific width before processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Defines the height of the crop area in pixels. This parameter is used to crop the image to a specific height before processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width for the output image after processing. It allows for resizing the image to a desired width.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for the output image after processing. It allows for resizing the image to a desired height.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - The global textual description to be encoded. This input is crucial for generating the corresponding visual representations and understanding the content described.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The CLIP model instance used for encoding the text. It plays a vital role in processing the text input and converting it into a format suitable for image generation or manipulation tasks.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`text_l`**
    - The local textual description to be encoded. This input provides additional detail or context to the global description, enhancing the specificity of the generated or manipulated image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output of the node, which includes the encoded text along with additional information necessary for image generation or manipulation tasks.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - Reroute
    - [ToBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ToBasicPipe.md)
    - KSampler (Efficient)
    - [Prompts Everywhere](../../cg-use-everywhere/Nodes/Prompts Everywhere.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - KRestartSamplerAdv



## Source code
```python
class CLIPTextEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
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
