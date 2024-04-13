# PNG Save Node
## Documentation
- Class name: `PNG_Save`
- Category: `Ebsynth Nodes`
- Output node: `True`

The PNG Save Node is designed to take an image or a batch of images and save them to a specified location in PNG format. It handles the conversion and encoding of images using the PNG encoding provided by torchvision's encode_png, ensuring the images are saved with the desired quality and naming conventions. The node also supports overwrite control for existing files.
## Input types
### Required
- **`save_path`**
    - Specifies the directory where the image(s) will be saved. It plays a crucial role in determining the output location of the saved images.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`file_name`**
    - Determines the base name for the saved image file(s). It influences the naming convention of the output files, allowing for easy identification and organization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`images`**
    - The image or batch of images to be saved. This input is central to the node's operation, as it provides the actual content to be encoded and saved in PNG format.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`quality`**
    - Sets the quality of the saved PNG image(s), ranging from 0 to 100. This parameter allows for fine-tuning of the output image's visual fidelity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`overwrite_mode`**
    - Controls whether existing files with the same name should be overwritten. This parameter ensures flexibility in managing file storage and avoiding unintended data loss.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PNG_Save:
    """
    PNG Save Node

    This node takes in an image and saves it to the specified location using the PNG encoding provided by torchvision's encode_png.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_path": ("STRING", {"default": "./ComfyUI/output"}),
                "file_name": ("STRING", {"default": "ComfyUI"}),
                "images": ("IMAGE",),
                "quality": ("INT", {"default": 96, "min": 0, "max": 100, "step": 1}),
                "overwrite_mode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = "Ebsynth Nodes"

    def save_image(self, save_path, file_name, images, quality, overwrite_mode=True):
        # If the images tensor has 4 dimensions, it's a batch. Otherwise, make it a batch of 1.
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        for idx, image in enumerate(images):
            # Construct full path for each image in the batch
            base_name, ext = os.path.splitext(file_name)
            # Ensure the extension is .png
            ext = ".png"
            current_file_name = f"{base_name}{ext}"
            full_path = os.path.join(save_path, current_file_name)

            # Ensure the directory exists
            directory = os.path.dirname(full_path)
            os.makedirs(directory, exist_ok=True)

            # Ensure the image tensor has the correct shape
            image_tensor = image.permute(2, 0, 1).float()  # Change to (C, H, W) format
            C, H, W = image_tensor.shape

            if C != 1 and C != 3:
                raise ValueError(f"Image tensor has {C} channels. Expected 1 or 3.")

            # Convert and encode the image to PNG
            image_tensor = image_tensor.mul(255).clamp(0, 255).type(torch.uint8)
            encoded_image = encode_png(image_tensor.cpu()).numpy().tobytes()

            # Check if file exists and overwrite mode is False
            if os.path.exists(full_path) and not overwrite_mode:
                print(f"File {full_path} already exists and overwrite mode is disabled.")
                continue

            # Save the encoded image
            with open(full_path, 'wb') as f:
                f.write(encoded_image)

            print(f"Image saved to: {full_path}")
                
        return {"status": "Image(s) saved successfully"}

```
