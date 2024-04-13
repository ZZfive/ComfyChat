# FreeU
## Documentation
- Class name: `FreeU`
- Category: `model_patches`
- Output node: `False`

The FreeU node is designed to enhance the flexibility and performance of neural network models by dynamically adjusting the scaling of hidden layers based on their mean activation values. It employs Fourier filtering to manage spatial frequencies in the data, adapting to different computational resources by switching to CPU processing when necessary.
## Input types
### Required
- **`model`**
    - The neural network model to be enhanced. It is the core component that undergoes dynamic scaling and Fourier filtering adjustments.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`b1`**
    - A scaling factor for the model's hidden layers, contributing to the dynamic adjustment of the model's performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`b2`**
    - Another scaling factor for the model's hidden layers, working alongside b1 to fine-tune the model's dynamic scaling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s1`**
    - A scale parameter for the Fourier filtering process, influencing the handling of spatial frequencies in the data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s2`**
    - Another scale parameter for the Fourier filtering, complementing s1 to optimize the spatial frequency management.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced neural network model, featuring dynamically scaled hidden layers and optimized spatial frequency handling through Fourier filtering.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - UltimateSDUpscale
    - [Bus Node](../../was-node-suite-comfyui/Nodes/Bus Node.md)
    - DynamicThresholdingFull
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)



## Source code
```python
class FreeU:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "b1": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "b2": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
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
                h[:,:h.shape[1] // 2] = h[:,:h.shape[1] // 2] * scale[0]
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
