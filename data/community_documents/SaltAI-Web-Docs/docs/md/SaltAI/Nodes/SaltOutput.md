# Salt Flow Output
## Documentation
- Class name: `SaltOutput`
- Category: `SALT/IO`
- Output node: `True`

The SaltOutput node is designed for handling various types of output data within a workflow, including images and strings. It supports a range of output formats and is capable of generating assets for multimedia content, making it a versatile component for output management in data processing pipelines.
## Input types
### Required
- **`output_name`**
    - Specifies the name of the output, which is used for identification and file naming purposes. It plays a crucial role in organizing and accessing the generated outputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_desc`**
    - Provides a description for the output, aiding in the contextual understanding and documentation of the generated data or asset.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_type`**
    - Determines the format of the output, supporting a variety of types including image formats and strings. This parameter is essential for correctly processing and rendering the output data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`output_data`**
    - Contains the actual data to be outputted, which can be of various types depending on the specified output_type. This parameter is central to the node's functionality, as it directly influences the output generation.
    - Comfy dtype: `*`
    - Python dtype: `Union[torch.Tensor, str]`
### Optional
- **`animation_fps`**
    - Specifies the frames per second for animated outputs, allowing control over the animation speed. This parameter is relevant for outputs that are animations, enhancing their visual quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`animation_quality`**
    - Defines the quality of the animation, offering options between default and high. This parameter allows for optimization of the output based on desired quality levels.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": ("STRING", {}),
                "output_desc": ("STRING", {}),
                "output_type": (
                    ["PNG", "JPEG", "GIF", "WEBP", "AVI", "MP4", "WEBM", "STRING"],
                ),
                "output_data": (WILDCARD, {}),
            },
            "optional": {
                "animation_fps": ("INT", {"min": 1, "max": 60, "default": 8}),
                "animation_quality": (["DEFAULT", "HIGH"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "output_subdir": None},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()

    FUNCTION = "output"
    CATEGORY = "SALT/IO"

    def output(
        self,
        output_name,
        output_desc,
        output_type,
        output_data,
        animation_fps=8,
        animation_quality="DEFAULT",
        unique_id=0,
        output_subdir=None
    ):
        is_asset = False
        asset_id = str(uuid.uuid4())

        # Determine if valid type
        if output_type.strip() == "" or output_type not in [
            "GIF",
            "WEBP",
            "AVI",
            "MP4",
            "WEBM",
        ]:
            if isinstance(output_data, torch.Tensor):
                output_type = "JPEG" if output_type == "JPEG" else "PNG"
            elif isinstance(output_data, str):
                output_type = "STRING"
            else:
                raise ValueError(
                    "Unsupported `output_type` supplied. Please provide `IMAGE` or `STRING` input."
                )

        # Is asset? I may have misunderstood this part
        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"]:
            is_asset = True

        # Determine output name, and sanitize if input (for filesystem)
        if output_name.strip() == "":
            output_name = "output_" + str(unique_id)
        else:
            output_name = sanitize_filename(output_name)

        # Create output dir based on uuid4
        subfolder = os.path.join(output_subdir or '', asset_id)
        output_path = os.path.join(folder_paths.get_output_directory(), subfolder)

        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            print(f"[SALT] Unable to create output directory `{output_path}`")

        results = []
        if output_type in ("PNG", "JPEG"):
            # Save all images in the tensor batch as specified by output_type
            try:
                for index, img in enumerate(output_data):
                    pil_image = tensor2pil(img)
                    file_prefix = output_name.strip().replace(" ", "_")
                    file_ext = f".{output_type.lower()}"
                    filename = f"{file_prefix}_{index:04d}{file_ext}"
                    image_path = os.path.join(output_path, filename)
                    pil_image.save(image_path, output_type)
                    results.append({
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    if os.path.exists(image_path):
                        print(f"[SALT] Saved image to `{image_path}`")
                    else:
                        print(f"[SALT] Unable to save image to `{image_path}`")

            except Exception as e:
                raise e

        if output_type in ["GIF", "WEBP", "AVI", "MP4", "WEBM"]:
            # Save animation file
            filename = os.path.join(output_path, f"{output_name}.{output_type.lower()}")
            animator = ImageAnimator(
                output_data, fps=int(animation_fps), quality=animation_quality
            )
            animator.save_animation(filename, format=output_type)
            results.append({
                "filename": os.path.basename(filename),
                "subfolder": subfolder,
                "type": "output"
            })
            if os.path.exists(filename):
                print(f"[SALT] Saved file to `{filename}`")
            else:
                print(f"[SALT] Unable to save file to `{filename}`")
        else:
            # Prepare output string
            if output_type == "STRING":
                results.append(str(output_data))

        # Output Dictionary
        ui = {
            "ui": {
                "salt_metadata": [
                    {
                        "salt_id": unique_id,
                        "salt_reference_uuid": asset_id,
                        "salt_description": output_desc,
                        "salt_asset": is_asset,
                        "salt_file_extension": output_type,
                    }
                ],
                "salt_output": results
            }
        }

        # Assign images for previews of supported types
        if output_type in ["PNG", "GIF", "WEBP", "JPEG"] and results:
            ui["ui"].update({"images": results})

        # Print to log
        print(f"[SaltOutput_{unique_id}] Output:")
        from pprint import pprint

        pprint(ui, indent=4)

        return ui

```
