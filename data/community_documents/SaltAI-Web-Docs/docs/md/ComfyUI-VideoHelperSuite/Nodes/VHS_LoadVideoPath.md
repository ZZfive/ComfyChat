# Load Video (Path) 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadVideoPath`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

This node is designed for loading video files from a specified path. It facilitates the process of importing video content into the system for further processing or analysis, making it a crucial component for workflows that involve video manipulation or analysis.
## Input types
### Required
- **`video`**
    - Specifies the path to the video file that needs to be loaded. It is essential for locating and accessing the video content for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`force_rate`**
    - Sets the frame rate to be forced on the loaded video, allowing for consistent frame rate across different videos.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`force_size`**
    - Determines the resolution to which the video should be resized during loading. This parameter is crucial for standardizing video dimensions across different processing steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`custom_width`**
    - Specifies the custom width to which the video should be resized if 'Custom Width' or 'Custom' force_size options are selected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - Specifies the custom height to which the video should be resized if 'Custom Height' or 'Custom' force_size options are selected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_load_cap`**
    - Limits the number of frames to be loaded from the video, useful for processing long videos or for sampling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_frames`**
    - Skips a specified number of frames at the beginning of the video, useful for bypassing unneeded content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_every_nth`**
    - Loads every nth frame from the video, allowing for frame rate reduction or selective frame processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - Returns the loaded video frames as images, facilitating further image-based processing or analysis.
    - Python dtype: `List[Image]`
- **`frame_count`**
    - Comfy dtype: `INT`
    - Provides the total number of frames loaded from the video, useful for understanding the video's length and for iteration purposes.
    - Python dtype: `int`
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - Returns the audio track extracted from the video, enabling separate audio processing or analysis.
    - Python dtype: `VHS_AUDIO`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"default": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                 "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                 "custom_width": ("INT", {"default": 512, "min": 0, "step": 8}),
                 "custom_height": ("INT", {"default": 512, "min": 0, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", )
    RETURN_NAMES = ("IMAGE", "frame_count", "audio",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)

```
