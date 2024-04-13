# VRAM Debug
## Documentation
- Class name: `VRAM_Debug`
- Category: `KJNodes`
- Output node: `False`

The VRAM_Debug node is designed to monitor and manage GPU memory usage during model operations. It provides insights into the available memory before and after specific operations, allowing for the optimization of memory usage by optionally clearing the CUDA cache and unloading model clones. This functionality is crucial for maintaining efficient GPU resource utilization, especially in environments with limited memory.
## Input types
### Required
- **`model`**
    - The model parameter represents the deep learning model whose memory usage is being monitored. It is crucial for identifying the specific model instance involved in the operation.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`empty_cuda_cache`**
    - This boolean parameter determines whether the CUDA cache should be emptied to free up GPU memory. It plays a significant role in managing memory usage, especially in memory-constrained environments.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`clip_vision`**
    - An optional parameter specifying a CLIP vision model to be unloaded, further aiding in memory management by releasing resources associated with model clones.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.nn.Module`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the original model passed as input, allowing for further operations or analysis.
    - Python dtype: `torch.nn.Module`
- **`freemem_before`**
    - Comfy dtype: `INT`
    - Provides the amount of free GPU memory available before the execution of the node's operations.
    - Python dtype: `int`
- **`freemem_after`**
    - Comfy dtype: `INT`
    - Indicates the amount of free GPU memory available after the node's operations, including any memory freed by clearing the CUDA cache or unloading model clones.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VRAM_Debug:
    
    @classmethod
    
    def INPUT_TYPES(s):
      return {
        "required": {
			  "model": ("MODEL",),
              "empty_cuda_cache": ("BOOLEAN", {"default": False}),
		  },
        "optional": {
            "clip_vision": ("CLIP_VISION", ),
        }
	  }
    RETURN_TYPES = ("MODEL", "INT", "INT",)
    RETURN_NAMES = ("model", "freemem_before", "freemem_after")
    FUNCTION = "VRAMdebug"
    CATEGORY = "KJNodes"

    def VRAMdebug(self, model, empty_cuda_cache, clip_vision=None):
        freemem_before = comfy.model_management.get_free_memory()
        print(freemem_before)
        if empty_cuda_cache:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if clip_vision is not None:
            print("unloading clip_vision_clone")
            comfy.model_management.unload_model_clones(clip_vision.patcher)
        freemem_after = comfy.model_management.get_free_memory()
        print(freemem_after)
        return (model, freemem_before, freemem_after)

```
