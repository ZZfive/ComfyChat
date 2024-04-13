# Load Video (Upload) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadVideo`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The VHS_LoadVideo node is designed for uploading and processing video files. It allows users to specify various parameters such as frame rate and size to customize the video processing according to their needs. This node is essential for preparing video data for further analysis or manipulation within the Video Helper Suite.
## Input types
### Required
- **`video`**
    - Specifies the path to the video file to be processed. It is the primary input for the video processing operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`force_rate`**
    - Determines the frame rate at which the video should be processed. This allows for standardization or modification of the video's frame rate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`force_size`**
    - Defines the desired resolution for the video output. This parameter allows users to resize the video according to specific requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`custom_width`**
    - Sets the custom width for the video output, enabling precise control over the video's dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - Sets the custom height for the video output, providing the ability to tailor the video's dimensions to specific needs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_load_cap`**
    - Limits the number of frames to be processed from the video. This is useful for handling large videos or focusing on specific segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_frames`**
    - Skips a specified number of frames at the beginning of the video. This can be used to bypass unneeded introductory content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Selects every Nth frame from the video for processing, allowing for sampling of the video content at a reduced rate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - Returns the processed video frames as images.
    - Python dtype: `List[Image]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - Provides the total number of frames processed from the video.
    - Python dtype: `int`
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - Returns the audio extracted from the video, if any.
    - Python dtype: `VHS_AUDIO`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - [CannyEdgePreprocessor](../../comfyui_controlnet_aux/Nodes/CannyEdgePreprocessor.md)
    - [ImageScaleToTotalPixels](../../Comfy/Nodes/ImageScaleToTotalPixels.md)
    - [ImageResize+](../../ComfyUI_essentials/Nodes/ImageResize+.md)
    - [TilePreprocessor](../../comfyui_controlnet_aux/Nodes/TilePreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)



## Source code
```python
class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                     "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     "custom_width": ("INT", {"default": 512, "min": 0, "step": 8}),
                     "custom_height": ("INT", {"default": 512, "min": 0, "step": 8}),
                     "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                     "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                     "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                     },}

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", )
    RETURN_NAMES = ("IMAGE", "frame_count", "audio",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(kwargs['video'].strip("\""))
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

```
