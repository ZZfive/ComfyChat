# StableCascade_StageB_Conditioning
## Documentation
- Class name: `StableCascade_StageB_Conditioning`
- Category: `_for_testing/stable_cascade`
- Output node: `False`

This node is designed to set a prior for the conditioning process in a stable cascade model, integrating latent stage information into the conditioning data to influence subsequent model stages.
## Input types
### Required
- **`conditioning`**
    - The conditioning data to be modified with the latent stage information, affecting the model's behavior in later stages.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`stage_c`**
    - The latent stage information to be integrated into the conditioning, influencing the model's subsequent stages.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning data, now containing latent stage information for influencing later model stages.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StableCascade_StageB_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "conditioning": ("CONDITIONING",),
                              "stage_c": ("LATENT",),
                             }}
    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "set_prior"

    CATEGORY = "_for_testing/stable_cascade"

    def set_prior(self, conditioning, stage_c):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c['samples']
            n = [t[0], d]
            c.append(n)
        return (c, )

```
