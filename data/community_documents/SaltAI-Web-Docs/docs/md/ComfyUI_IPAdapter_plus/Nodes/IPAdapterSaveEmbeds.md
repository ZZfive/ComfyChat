# Save IPAdapter Embeds
## Documentation
- Class name: `IPAdapterSaveEmbeds`
- Category: `ipadapter`
- Output node: `True`

The IPAdapterSaveEmbeds node is designed for persisting IPAdapter embeddings to disk. It facilitates the storage of embeddings with a customizable filename prefix, ensuring that the embeddings can be easily organized and retrieved for future use.
## Input types
### Required
- **`embeds`**
    - The 'embeds' parameter represents the embeddings to be saved. It is crucial for specifying the data that will be persisted to disk, affecting the node's execution by determining the content of the output file.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `torch.Tensor`
- **`filename_prefix`**
    - The 'filename_prefix' parameter allows for customization of the saved file's name, providing a means to organize and easily identify the embeddings on disk. It influences the naming convention of the output file, aiding in the retrieval of specific embeddings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterSaveEmbeds:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "embeds/IPAdapter"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )

```
