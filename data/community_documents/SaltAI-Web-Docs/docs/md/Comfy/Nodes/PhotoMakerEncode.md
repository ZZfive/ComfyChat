# PhotoMakerEncode
## Documentation
- Class name: `PhotoMakerEncode`
- Category: `_for_testing/photomaker`
- Output node: `False`

The PhotoMakerEncode node is designed to integrate visual information from images with textual prompts to enhance the generation of image embeddings. It leverages a CLIP model for vision tasks, applies a series of projections and fusions to the image embeddings, and combines them with prompt embeddings to produce updated embeddings that are more contextually relevant to the given text and images.
## Input types
### Required
- **`photomaker`**
    - Represents the model used for encoding and integrating visual information from images into the text embeddings. It is essential for the fusion process that combines visual and textual information.
    - Comfy dtype: `PHOTOMAKER`
    - Python dtype: `PhotoMakerIDEncoder`
- **`image`**
    - The image input that provides visual information to be encoded and fused with text embeddings. It plays a crucial role in enhancing the contextual relevance of the generated embeddings.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clip`**
    - The CLIP model used for processing the text and potentially the image. It is instrumental in generating text embeddings and may also contribute to image processing.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIPModel`
- **`text`**
    - The textual prompt that will be enhanced with visual information from the image. It guides the generation of contextually relevant embeddings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The updated prompt embeddings after fusion with image embeddings. These embeddings are more contextually enriched, combining both textual and visual information.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PhotoMakerEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "photomaker": ("PHOTOMAKER",),
                              "image": ("IMAGE",),
                              "clip": ("CLIP", ),
                              "text": ("STRING", {"multiline": True, "default": "photograph of photomaker"}),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_photomaker"

    CATEGORY = "_for_testing/photomaker"

    def apply_photomaker(self, photomaker, image, clip, text):
        special_token = "photomaker"
        pixel_values = comfy.clip_vision.clip_preprocess(image.to(photomaker.load_device)).float()
        try:
            index = text.split(" ").index(special_token) + 1
        except ValueError:
            index = -1
        tokens = clip.tokenize(text, return_word_ids=True)
        out_tokens = {}
        for k in tokens:
            out_tokens[k] = []
            for t in tokens[k]:
                f = list(filter(lambda x: x[2] != index, t))
                while len(f) < len(t):
                    f.append(t[-1])
                out_tokens[k].append(f)

        cond, pooled = clip.encode_from_tokens(out_tokens, return_pooled=True)

        if index > 0:
            token_index = index - 1
            num_id_images = 1
            class_tokens_mask = [True if token_index <= i < token_index+num_id_images else False for i in range(77)]
            out = photomaker(id_pixel_values=pixel_values.unsqueeze(0), prompt_embeds=cond.to(photomaker.load_device),
                            class_tokens_mask=torch.tensor(class_tokens_mask, dtype=torch.bool, device=photomaker.load_device).unsqueeze(0))
        else:
            out = cond

        return ([[out, {"pooled_output": pooled}]], )

```
