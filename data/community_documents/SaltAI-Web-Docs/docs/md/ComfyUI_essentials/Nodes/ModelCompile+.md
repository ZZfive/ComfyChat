# 🔧 Compile Model
## Documentation
- Class name: `ModelCompile+`
- Category: `essentials`
- Output node: `False`

This node compiles a given model for optimized execution, allowing for dynamic adjustments and full graph compilation based on the specified mode. It aims to enhance model performance by leveraging PyTorch's compilation capabilities.
## Input types
### Required
- **`model`**
    - The model to be compiled for optimized execution. It is the primary subject of the compilation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`fullgraph`**
    - Indicates whether the full computational graph of the model should be considered during compilation, potentially enhancing optimization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`dynamic`**
    - Determines if dynamic shapes and operations within the model are allowed during compilation, offering flexibility in handling variable input sizes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mode`**
    - Specifies the compilation mode, which can range from default settings to aggressive autotuning and optimization strategies, affecting the compilation's thoroughness and performance impact.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The compiled model, optimized for improved performance and efficiency.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ModelCompile():  
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fullgraph": ("BOOLEAN", { "default": False }),
                "dynamic": ("BOOLEAN", { "default": False }),
                "mode": (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],),
            },
        }
    
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, model, fullgraph, dynamic, mode):
        work_model = model.clone()
        torch._dynamo.config.suppress_errors = True
        work_model.model.diffusion_model = torch.compile(work_model.model.diffusion_model, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
        return( work_model, )

```
