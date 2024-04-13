# DummyLatentOut
## Documentation
- Class name: `DummyLatentOut`
- Category: `KJNodes`
- Output node: `True`

The DummyLatentOut node is designed to pass through the input latent representation without any modification. It serves as a placeholder or a no-operation (noop) node within a pipeline, ensuring the flow of data without altering it.
## Input types
### Required
- **`latent`**
    - The 'latent' parameter represents the latent representation to be passed through. It is crucial for the node's operation as it directly influences the output, which is an unaltered version of this input.
    - Comfy dtype: `LATENT`
    - Python dtype: `Tuple[torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is the unmodified latent representation received as input, ensuring the data flows through the pipeline without alteration.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DummyLatentOut:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "dummy"
    CATEGORY = "KJNodes"
    OUTPUT_NODE = True

    def dummy(self, latent):

        return (latent,)

```
