# Save Images No Display (Mikey)
## Documentation
- Class name: `Save Images No Display`
- Category: `Mikey/Image`
- Output node: `True`

This node is designed to save images to a specified directory without displaying them in the UI. It inherits functionality from a parent class to perform the image saving operation, focusing on backend storage while omitting any form of visual feedback or display.
## Input types
### Required
- **`images`**
    - The images to be saved. This parameter is crucial for determining which images are processed and stored by the node.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`sub_directory`**
    - Specifies the sub-directory within the main directory where the images will be saved. This helps organize saved images into specific categories or groups.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filename_text_1`**
    - The first part of the filename for the saved images. This can be used to add specific identifiers or tags to filenames.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filename_text_2`**
    - The second part of the filename for the saved images, allowing for further customization and organization of saved files.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filename_text_3`**
    - The third part of the filename for the saved images, providing additional options for file naming and organization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filename_separator`**
    - A character or string used to separate different parts of the filename, enhancing readability and structure.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`timestamp`**
    - Indicates whether a timestamp should be included in the filename, helping to timestamp and version saved images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`counter_type`**
    - Specifies the type of counter to be used in the filename, aiding in the sequential numbering of saved images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`filename_text_1_pos`**
    - The position of the first filename text within the overall filename structure, allowing for customizable filename formats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`filename_text_2_pos`**
    - The position of the second filename text within the filename, enabling further customization of the filename structure.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`filename_text_3_pos`**
    - The position of the third filename text in the filename, offering additional flexibility in naming conventions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`timestamp_pos`**
    - The position of the timestamp within the filename, ensuring that timestamps are placed according to specific formatting requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`timestamp_type`**
    - Defines the format of the timestamp included in the filename, allowing for consistent time-based tagging of images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`counter_pos`**
    - The position of the counter in the filename, facilitating orderly and sequential naming of saved images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`extra_metadata`**
    - Additional metadata to be included with the saved images, enriching the information stored alongside the images.
    - Comfy dtype: `STRING`
    - Python dtype: `Dict[str, Any]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveImageNoDisplay(SaveImagesMikeyML):
    # inherits from SaveImagesMikeyML
    # only difference is we are not going to output anything to the UI
    def __init__(self):
        super().__init__()

    RETURN_TYPES = ()
    FUNCTION = "save_images_no_display"
    OUTPUT_NODE = True
    CATEGORY = "Mikey/Image"

    def save_images_no_display(self, images, sub_directory, filename_text_1, filename_text_2, filename_text_3,
                    filename_separator, timestamp, counter_type,
                    filename_text_1_pos, filename_text_2_pos, filename_text_3_pos,
                    timestamp_pos, timestamp_type, counter_pos, extra_metadata,
                    prompt=None, extra_pnginfo=None):
        self.save_images(images, sub_directory, filename_text_1, filename_text_2, filename_text_3,
                    filename_separator, timestamp, counter_type,
                    filename_text_1_pos, filename_text_2_pos, filename_text_3_pos,
                    timestamp_pos, timestamp_type, counter_pos, extra_metadata,
                    prompt, extra_pnginfo)
        return (None,)

```
