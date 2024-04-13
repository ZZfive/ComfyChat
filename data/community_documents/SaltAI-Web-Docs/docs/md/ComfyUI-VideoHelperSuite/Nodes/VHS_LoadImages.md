# Load Images (Upload) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadImages`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The VHS_LoadImages node is designed for loading a batch of images from a specified directory. It supports filtering and limiting the number of images loaded, making it a versatile tool for managing and preprocessing image datasets for video processing tasks.
## Input types
### Required
- **`directory`**
    - Specifies the directory from which images are to be loaded. It is crucial for identifying the source of the images to be processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_load_cap`**
    - Limits the number of images to be loaded from the directory. This parameter is useful for controlling the dataset size or for testing purposes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_images`**
    - Skips a specified number of images from the beginning of the directory. This can be used for excluding initial images that are not needed for processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every Nth image from the directory, allowing for subsampling of the dataset. This is useful for reducing the dataset size or for selecting a representative subset of images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The loaded images as a batch, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - Generated masks for the loaded images, useful for image segmentation or editing tasks.
    - Python dtype: `torch.Tensor`
- **`int`**
    - Comfy dtype: `INT`
    - The total number of images loaded, providing insight into the dataset size after applying the loading parameters.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - Reroute
    - [ScribblePreprocessor](../../comfyui_controlnet_aux/Nodes/ScribblePreprocessor.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)
    - [TilePreprocessor](../../comfyui_controlnet_aux/Nodes/TilePreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [Zoe-DepthMapPreprocessor](../../comfyui_controlnet_aux/Nodes/Zoe-DepthMapPreprocessor.md)



## Source code
```python
class LoadImagesFromDirectoryUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return validate_load_images(directory)

```
