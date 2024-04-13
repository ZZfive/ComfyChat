# Load IPAdapter Embeds
## Documentation
- Class name: `IPAdapterLoadEmbeds`
- Category: `ipadapter`
- Output node: `False`

This node is designed for loading pre-saved IPAdapter embeddings from a specified directory. It facilitates the retrieval of embeddings for further processing or analysis within the IPAdapter framework.
## Input types
### Required
- **`embeds`**
    - Specifies the file names of the embeddings to be loaded. This allows for selective loading of embeddings based on the user's requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`embeds`**
    - Comfy dtype: `EMBEDS`
    - Returns the loaded embeddings as tensors, ready for use in subsequent IPAdapter operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterLoadEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [os.path.relpath(os.path.join(root, file), input_dir) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith('.ipadpt')]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter"

    def load(self, embeds):
        path = folder_paths.get_annotated_filepath(embeds)
        output = torch.load(path).cpu()

        return (output, )

```
