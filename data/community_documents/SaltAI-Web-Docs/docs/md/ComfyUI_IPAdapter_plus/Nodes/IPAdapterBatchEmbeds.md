# IPAdapter Batch Embeds
## Documentation
- Class name: `IPAdapterBatchEmbeds`
- Category: `ipadapter`
- Output node: `False`

The IPAdapterBatchEmbeds node is designed for the purpose of combining two sets of embeddings into a single batch. This operation is crucial for scenarios where embeddings from different sources or modalities need to be processed together, enhancing the flexibility and applicability of embedding-based workflows.
## Input types
### Required
- **`embed1`**
    - The first set of embeddings to be combined. It plays a crucial role in the batch operation by providing one half of the data that will be concatenated.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
- **`embed2`**
    - The second set of embeddings to be combined. It complements the first set of embeddings, enabling the creation of a unified batch for further processing.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
## Output types
- **`embeds`**
    - Comfy dtype: `EMBEDS`
    - The combined set of embeddings resulting from concatenating the input embeddings along a specified dimension. This unified batch is ready for subsequent processing or analysis steps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterBatchEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embed1": ("EMBEDS",),
            "embed2": ("EMBEDS",),
        }}

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "batch"
    CATEGORY = "ipadapter"

    def batch(self, embed1, embed2):
        return (torch.cat((embed1, embed2), dim=1), )

```
