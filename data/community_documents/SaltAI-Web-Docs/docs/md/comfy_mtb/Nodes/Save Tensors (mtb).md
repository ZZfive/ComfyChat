# Save Tensors (mtb)
## Documentation
- Class name: `Save Tensors (mtb)`
- Category: `mtb/debug`
- Output node: `True`

The Save Tensors node is designed for persisting torch tensors, such as images, masks, or latent representations, to disk. This functionality is particularly useful for debugging purposes, allowing for the examination of these tensors outside of the Comfy environment.
## Input types
### Required
- **`filename_prefix`**
    - Specifies the prefix for the filenames under which the tensors will be saved. This prefix helps in organizing and identifying saved tensors.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`image`**
    - An optional image tensor to be saved. Useful for storing image data for later analysis or visualization.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[torch.Tensor]`
- **`mask`**
    - An optional mask tensor to be saved. Masks are typically used for segmenting or highlighting specific parts of an image.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
- **`latent`**
    - An optional latent tensor to be saved. Latent tensors represent high-dimensional data in a compressed form, useful for deep learning models.
    - Comfy dtype: `LATENT`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveTensors:
    """Save torch tensors (image, mask or latent) to disk, useful to debug things outside comfy"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "mtb/debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "ComfyPickle"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "latent": ("LATENT",),
            },
        }

    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "mtb/debug"

    def save(
        self,
        filename_prefix,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        full_output_folder = Path(full_output_folder)
        if image is not None:
            image_file = f"{filename}_image_{counter:05}.pt"
            torch.save(image, full_output_folder / image_file)
            # np.save(full_output_folder/ image_file, image.cpu().numpy())

        if mask is not None:
            mask_file = f"{filename}_mask_{counter:05}.pt"
            torch.save(mask, full_output_folder / mask_file)
            # np.save(full_output_folder/ mask_file, mask.cpu().numpy())

        if latent is not None:
            # for latent we must use pickle
            latent_file = f"{filename}_latent_{counter:05}.pt"
            torch.save(latent, full_output_folder / latent_file)
            # pickle.dump(latent, open(full_output_folder/ latent_file, "wb"))

            # np.save(full_output_folder/ latent_file, latent[""].cpu().numpy())

        return f"{filename_prefix}_{counter:05}"

```
