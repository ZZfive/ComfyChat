# Checkpoint Loader Simple (Mikey)
## Documentation
- Class name: `Checkpoint Loader Simple Mikey`
- Category: `Mikey/Loaders`
- Output node: `False`

This node is designed to load a specific checkpoint from a predefined directory, handling the process of locating the checkpoint file, loading it along with optional components like VAE and CLIP models, and extracting relevant metadata such as the checkpoint's name and hash. It simplifies the checkpoint loading process by abstracting away the configuration details and directly providing the necessary outputs for further processing.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the name of the checkpoint to be loaded. This parameter is crucial as it determines which checkpoint file will be accessed and loaded from the directory.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[str]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The loaded model from the checkpoint.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model loaded alongside the main model, if requested.
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model loaded alongside the main model, if requested.
    - Python dtype: `torch.nn.Module`
- **`ckpt_name`**
    - Comfy dtype: `STRING`
    - The name of the loaded checkpoint, extracted from the file path.
    - Python dtype: `str`
- **`ckpt_hash`**
    - Comfy dtype: `STRING`
    - A hash of the checkpoint file, used for verification or tracking purposes.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CheckpointLoaderSimpleMikey:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "ckpt_name", "ckpt_hash")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Mikey/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True, unique_id=None, extra_pnginfo=None, prompt=None):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # get hash using python, there is no function in the comfy library
        hash = get_file_hash(ckpt_path)[:10]
        # just return the filename, not path
        ckpt_name = os.path.basename(ckpt_name)
        prompt.get(str(unique_id))['inputs']['output_ckpt_hash'] = hash
        prompt.get(str(unique_id))['inputs']['output_ckpt_name'] = ckpt_name
        return out[:3] + (ckpt_name, hash)

```
