# 🔧 Mask Preview
## Documentation
- Class name: `MaskPreview+`
- Category: `essentials`
- Output node: `True`

The MaskPreview node is designed for generating a preview of masks by reshaping and expanding the mask tensor for visualization purposes. It temporarily saves these visualizations with a unique prefix to a designated output directory.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the mask tensor to be visualized. It is crucial for generating the preview image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)

```
