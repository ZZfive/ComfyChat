# FreeU_V2
## Documentation
- Class name: `FreeU_V2`
- Category: `model_patches`
- Output node: `False`

The FreeU_V2 node is designed to enhance the flexibility and performance of neural network models by dynamically adjusting the scaling of hidden layers based on the model's internal state. This adjustment is made possible through the application of a unique scaling mechanism that takes into account the mean, maximum, and minimum values of the hidden layers, thereby optimizing the model's learning and inference capabilities.
## Input types
### Required
- **`model`**
    - The neural network model to be enhanced. This parameter is crucial as it determines the base architecture that the FreeU_V2 node will operate on, directly influencing the effectiveness of the dynamic scaling mechanism.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`b1`**
    - A scaling factor that influences the adjustment of the model's hidden layers, contributing to the dynamic scaling mechanism's ability to optimize the model's performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b2`**
    - Another scaling factor that works in conjunction with b1 to fine-tune the adjustment of the model's hidden layers, enhancing the dynamic scaling mechanism's effectiveness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s1`**
    - A scale parameter that further refines the dynamic adjustment of the model's hidden layers, aiding in the optimization of the model's learning and inference processes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s2`**
    - A scale parameter that complements s1, contributing to the precise tuning of the model's hidden layers for improved performance and flexibility.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced neural network model with dynamically scaled hidden layers, ready for improved learning and inference tasks.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - UltimateSDUpscale
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)
    - Reroute
    - [PatchModelAddDownscale](../../Comfy/Nodes/PatchModelAddDownscale.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [VideoLinearCFGGuidance](../../Comfy/Nodes/VideoLinearCFGGuidance.md)
    - IPAdapter
    - UltimateSDUpscaleNoUpscale



## Source code
```python
class FreeU_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "b2": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, b1, b2, s1, s2):
        model_channels = model.model.model_config.unet_config["model_channels"]
        scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
        on_cpu_devices = {}

        def output_block_patch(h, hsp, transformer_options):
            scale = scale_dict.get(h.shape[1], None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * ((scale[0] - 1 ) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except:
                        print("Device", hsp.device, "does not support the torch.fft functions used in the FreeU node, switching to CPU.")
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

            return h, hsp

        m = model.clone()
        m.set_model_output_block_patch(output_block_patch)
        return (m, )

```
