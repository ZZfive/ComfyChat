# ReferenceOnlySimple3
## Documentation
- Class name: `ReferenceOnlySimple3`
- Category: `KJNodes/experiments`
- Output node: `False`

This node is designed to manipulate and combine latent representations in a generative model, specifically by applying custom reference-based transformations. It allows for the cloning of a model's state, the adjustment of latent sizes based on input batch sizes, and the custom application of transformations to latent vectors. The node facilitates the integration of additional reference data into the generative process, enhancing the model's ability to generate outputs that are influenced by multiple reference inputs.
## Input types
### Required
- **`model`**
    - The generative model whose state is to be cloned and modified. This model serves as the foundation for applying the reference-based transformations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`reference`**
    - The primary reference data used to influence the generative process. It plays a crucial role in determining the size of the latent vectors and the initial batch adjustments.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`reference2`**
    - An additional set of reference data used alongside the primary reference to further influence the generative process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`input`**
    - The input latent representations to be transformed and combined with the reference data. This input is central to the node's functionality, enabling the generation of enhanced outputs.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`batch_size`**
    - The size of the batch for processing the latent vectors. It affects the resizing of latent dimensions and the overall batch processing within the node.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with custom transformations applied.
    - Python dtype: `torch.nn.Module`
- **`latent`**
    - Comfy dtype: `LATENT`
    - A dictionary containing the combined latent representations and an optional noise mask.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ReferenceOnlySimple3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "reference": ("LATENT",),
                              "reference2": ("LATENT",),
                              "input": ("LATENT",),
                               "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                               }}
    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "reference_only"

    CATEGORY = "KJNodes/experiments"
 
    def reference_only(self, model, reference, reference2, input, batch_size):
        model_reference = model.clone()
        size_latent = list(reference["samples"].shape)
        size_latent[0] = batch_size
        latent = input
 
        batch = latent["samples"].shape[0] + reference["samples"].shape[0] + reference2["samples"].shape[0]
  
        def reference_apply(q, k, v, extra_options):
            k = k.clone().repeat(1, 2, 1)
            offset = 0
            if q.shape[0] > batch:
                offset = batch
                
            re = extra_options["transformer_index"] % 2
 
            for o in range(0, q.shape[0], batch):
                for x in range(1, batch):
                    k[x + o, q.shape[1]:] = q[o + re,:]
            return q, k, k
 
        model_reference.set_model_attn1_patch(reference_apply)

        out_latent = torch.cat((reference["samples"], reference2["samples"], latent["samples"]))
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
        else:
            mask = torch.ones((64,64), dtype=torch.float32, device="cpu")
            mask = mask.repeat(latent["samples"].shape[0], 1, 1)
 
        out_mask = torch.zeros((1,mask.shape[1],mask.shape[2]), dtype=torch.float32, device="cpu")
        return (model_reference, {"samples": out_latent, "noise_mask": torch.cat((out_mask,out_mask, mask))})

```
