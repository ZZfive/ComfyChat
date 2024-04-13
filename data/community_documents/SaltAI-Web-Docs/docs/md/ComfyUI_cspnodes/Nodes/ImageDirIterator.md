# Image Dir Iterator
## Documentation
- Class name: `ImageDirIterator`
- Category: `cspnodes`
- Output node: `False`

The ImageDirIterator node is designed to iterate through images in a specified directory, allowing for the retrieval of images based on their index. This functionality is particularly useful for processing or analyzing a sequence of images in a structured manner, where the order of images can be determined by their modification time.
## Input types
### Required
- **`directory_path`**
    - Specifies the path to the directory containing the images to be iterated over. This path is crucial for locating and accessing the images for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`image_index`**
    - Determines the specific image to retrieve from the sorted list of images in the directory. The index is based on the sorting order by modification time, with 0 representing the most recently modified image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Returns the image at the specified index as a tensor, preprocessed and ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageDirIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {}),
                "image_index": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image_by_index"
    CATEGORY = "cspnodes"

    def get_image_by_index(self, directory_path, image_index):
        # Get list of image files sorted by modification time (most recent first)
        image_files = sorted(
            [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))],
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )

        # Validate index
        if image_index < 0 or image_index >= len(image_files):
            raise IndexError("Image index out of range.")

        # Load and preprocess the image
        image = Image.open(image_files[image_index])
        image = ImageOps.exif_transpose(image)  # Correct orientation
        image = image.convert("RGB")  # Ensure image is in RGB format

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

        return (image_tensor,)

```
