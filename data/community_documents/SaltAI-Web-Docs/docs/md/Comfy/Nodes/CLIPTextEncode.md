# CLIP Text Encode (Prompt)
## Documentation
- Class name: `CLIPTextEncode`
- Category: `conditioning`
- Output node: `False`

The CLIPTextEncode node is designed to encode textual inputs using a CLIP model, transforming text into a form that can be utilized for conditioning in generative tasks. It abstracts the complexity of text tokenization and encoding, providing a streamlined interface for generating text-based conditioning vectors.
## Input types
### Required
- **`text`**
    - The 'text' parameter is the textual input that will be encoded. It plays a crucial role in determining the output conditioning vector, as it is the primary source of information for the encoding process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The 'clip' parameter represents the CLIP model used for text tokenization and encoding. It is essential for converting the textual input into a conditioning vector, influencing the quality and relevance of the generated output.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output 'conditioning' is a vector representation of the input text, encoded by the CLIP model. It serves as a crucial component for guiding generative models in producing relevant and coherent outputs.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - KSampler //Inspire
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - Reroute
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - [ToBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ToBasicPipe.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)



## Source code
```python
class CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

```
