# Apply IPAdapter from Encoded
## Documentation
- Class name: `IPAdapterApplyEncoded`
- Category: `ipadapter`
- Output node: `False`

This node applies an IPAdapter to encoded embeddings, allowing for the modification and enhancement of the embeddings based on the IPAdapter's configuration. It supports adjusting the influence of the IPAdapter through weights and can operate over a specified range of the embedding.
## Input types
### Required
- **`ipadapter`**
    - The IPAdapter to be applied. It is crucial for modifying and enhancing the embeddings.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `torch.nn.Module`
- **`embeds`**
    - The encoded embeddings to which the IPAdapter will be applied. These embeddings are the target of the modification and enhancement.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
- **`model`**
    - The model used in conjunction with the IPAdapter and embeddings. It defines the context in which the embeddings are applied.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`weight`**
    - A float value that adjusts the influence of the IPAdapter on the embeddings. It allows for fine-tuning the effect of the adapter.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the method of applying the weight to the IPAdapter's influence on the embeddings. Options include 'original', 'linear', and 'channel penalty'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - A float value indicating the start point in the embedding to begin applying the IPAdapter. Allows for targeted modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - A float value indicating the end point in the embedding for applying the IPAdapter. Enables precise control over the modification range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - A boolean indicating whether to unfold the batch during processing. This affects how the embeddings are processed by the IPAdapter.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`attn_mask`**
    - An optional mask that can be applied during the IPAdapter's processing. It allows for selective attention in the embeddings.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model after the IPAdapter has been applied, reflecting the modifications and enhancements made to the embeddings.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [CR Apply LoRA Stack](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Apply LoRA Stack.md)
    - Reroute
    - [ADE_AnimateDiffLoaderWithContext](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_AnimateDiffLoaderWithContext.md)



## Source code
```python
class IPAdapterApplyEncoded(IPAdapterApply):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "embeds": ("EMBEDS",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "unfold_batch": ("BOOLEAN", { "default": False }),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

```
