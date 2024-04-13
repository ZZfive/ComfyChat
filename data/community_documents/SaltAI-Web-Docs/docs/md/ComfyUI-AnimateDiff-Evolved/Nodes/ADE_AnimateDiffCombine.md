# DO NOT USE, USE VideoCombine from ComfyUI-VideoHelperSuite instead! AnimateDiff Combine [DEPRECATED, DO NOT USE] 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffCombine`
- Category: ``
- Output node: `True`

The ADE_AnimateDiffCombine node was designed to facilitate the creation and manipulation of animated sequences by combining different elements or frames into a cohesive animation. It abstracts the complexities involved in the animation process, providing a streamlined workflow for users to generate dynamic content.
## Input types
### Required
- **`images`**
    - Represents the sequence of images to be combined into an animation. It plays a crucial role in determining the visual content of the final output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image.Image]`
- **`frame_rate`**
    - Specifies the playback speed of the animation, affecting how smooth or fast the animation appears.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`loop_count`**
    - Determines the number of times the animation will loop, with a specific focus on enabling repetitive playback.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`filename_prefix`**
    - Acts as the base name for the output file, allowing users to identify and organize their animations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`format`**
    - Defines the output format of the animation, influencing compatibility and quality. It is not a list but a single string value representing the chosen format.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`pingpong`**
    - Enables a back-and-forth playback mode, adding a dynamic effect to the animation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`save_image`**
    - Controls whether the first frame of the animation is saved as a separate image file, including metadata.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`gif`**
    - Comfy dtype: `GIF`
    - The output is an animated GIF file, encapsulating the combined frames into a single, loopable animation. This format is widely supported and allows for easy sharing and viewing.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffCombine_Deprecated:
    ffmpeg_warning_already_shown = False
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_path = shutil.which("ffmpeg")
        #Hide ffmpeg formats if ffmpeg isn't available
        if ffmpeg_path is not None:
            ffmpeg_formats = ["video/"+x[:-5] for x in folder_paths.get_filename_list(Folders.VIDEO_FORMATS)]
        else:
            ffmpeg_formats = []
            if not s.ffmpeg_warning_already_shown:
                # Deprecated node are now hidden, so no need to show warning unless node is used.
                # logger.warning("This warning can be ignored, you should not be using the deprecated AnimateDiff Combine node anyway. If you are, use Video Combine from ComfyUI-VideoHelperSuite instead. ffmpeg could not be found. Outputs that require it have been disabled")
                s.ffmpeg_warning_already_shown = True
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_image": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("GIF",)
    OUTPUT_NODE = True
    CATEGORY = ""
    FUNCTION = "generate_gif"

    def generate_gif(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_image=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        logger.warning("Do not use AnimateDiff Combine node, it is deprecated. Use Video Combine node from ComfyUI-VideoHelperSuite instead. Video nodes from VideoHelperSuite are actively maintained, more feature-rich, and also automatically attempts to get ffmpeg.")
        # convert images to numpy
        frames: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)
            
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, file)
        frames[0].save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        if pingpong:
            frames = frames + frames[-2:0:-1]
        
        format_type, format_ext = format.split("/")
        file = f"{filename}_{counter:05}_.{format_ext}"
        file_path = os.path.join(full_output_folder, file)
        if format_type == "image":
            # Use pillow directly to save an animated image
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
        else:
            # Use ffmpeg to save a video
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                #Should never be reachable
                raise ProcessLookupError("Could not find ffmpeg")

            video_format_path = folder_paths.get_full_path("video_formats", format_ext + ".json")
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{frames[0].width}x{frames[0].height}"
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + video_format['main_pass'] + [file_path]

            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}}

```
