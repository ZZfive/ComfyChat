# Perp-Neg
## Documentation
- Class name: `PerpNeg`
- Category: `_for_testing`
- Output node: `False`

The PerpNeg node is designed to modify a given model's behavior by adjusting its sampling process. It introduces a perpendicular negative component to the model's noise prediction, aiming to enhance the generation quality by mitigating the influence of undesired conditioning.
## Input types
### Required
- **`model`**
    - The model to be modified, providing the base for the node's operation. It's crucial for defining the starting point of the adjustment process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`empty_conditioning`**
    - An empty conditioning input used to generate a baseline for comparison, essential for calculating the perpendicular negative component.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `torch.Tensor`
- **`neg_scale`**
    - A scaling factor for the perpendicular negative component, allowing fine-tuning of its influence on the model's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with an updated sampling process, incorporating the perpendicular negative adjustment.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PerpNeg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "empty_conditioning": ("CONDITIONING", ),
                             "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, empty_conditioning, neg_scale):
        m = model.clone()
        nocond = comfy.sample.convert_cond(empty_conditioning)

        def cfg_function(args):
            model = args["model"]
            noise_pred_pos = args["cond_denoised"]
            noise_pred_neg = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            nocond_processed = comfy.samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

            (noise_pred_nocond, _) = comfy.samplers.calc_cond_uncond_batch(model, nocond_processed, None, x, sigma, model_options)

            pos = noise_pred_pos - noise_pred_nocond
            neg = noise_pred_neg - noise_pred_nocond
            perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
            perp_neg = perp * neg_scale
            cfg_result = noise_pred_nocond + cond_scale*(pos - perp_neg)
            cfg_result = x - cfg_result
            return cfg_result

        m.set_model_sampler_cfg_function(cfg_function)

        return (m, )

```
