# CLIPTextEncodeControlnet
## Documentation
- Class name: `CLIPTextEncodeControlnet`
- Category: `_for_testing/conditioning`
- Output node: `False`

This node is designed for encoding text inputs using the CLIP model to produce conditioning data tailored for control networks. It enhances the conditioning data with specific control signals derived from the encoded text, facilitating more precise and controlled generation processes.
## Input types
### Required
- **`clip`**
    - The CLIP model used for text tokenization and encoding. It plays a crucial role in converting text inputs into a format suitable for further processing and conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`conditioning`**
    - A list of conditioning data to be enhanced with control signals derived from the encoded text. It serves as the basis for applying text-based modifications to the generation process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
- **`text`**
    - The text input to be encoded. This text is tokenized and encoded using the CLIP model to generate control signals for the conditioning data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Enhanced conditioning data, augmented with control signals derived from the encoded text, ready for use in controlled generation processes.
    - Python dtype: `List[Tuple[Any, Dict[str, Any]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPTextEncodeControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP", ), "conditioning": ("CONDITIONING", ), "text": ("STRING", {"multiline": True})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "_for_testing/conditioning"

    def encode(self, clip, conditioning, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return (c, )

```
