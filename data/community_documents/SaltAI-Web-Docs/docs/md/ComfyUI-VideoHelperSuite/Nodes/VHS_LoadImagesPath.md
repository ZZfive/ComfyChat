# Load Images (Path) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadImagesPath`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

This node is designed for loading images from a specified directory path. It supports loading a batch of images, optionally applying constraints such as skipping initial images, capping the number of images loaded, and selecting every nth image to form a subset. This functionality is crucial for processing and manipulating image data in bulk, especially in workflows that involve image analysis, transformation, or integration with other media types.
## Input types
### Required
- **`directory`**
    - Specifies the directory path from which images are to be loaded. This parameter is essential for defining the source of the image data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`image_load_cap`**
    - Limits the number of images to be loaded from the directory. This parameter helps in managing resource utilization by controlling the batch size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_images`**
    - Skips a specified number of initial images in the directory. This can be useful for bypassing unwanted or irrelevant images at the beginning of the dataset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every nth image from the directory, allowing for the creation of a subset of images. This parameter is useful for sampling or reducing the dataset size while maintaining a representative selection of images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The loaded images, represented as tensors. This output is crucial for subsequent image processing or analysis tasks.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The masks associated with the loaded images, if available. Masks are useful for image segmentation and other tasks that require distinguishing parts of an image.
    - Python dtype: `torch.Tensor`
- **`int`**
    - Comfy dtype: `INT`
    - The total number of images loaded. This information can be useful for tracking dataset size or for conditional logic based on the number of images processed.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ImpactImageBatchToImageList](../../ComfyUI-Impact-Pack/Nodes/ImpactImageBatchToImageList.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [IPAdapterEncoder](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterEncoder.md)
    - LinearBatchCreativeInterpolation



## Source code
```python
class LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images", "vhs_path_extensions": []}),
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
        if directory is None or validate_load_images(directory) != True:
            raise Exception("directory is not valid: " + directory)

        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(directory)

```
