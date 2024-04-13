# IFRNet VFI
## Documentation
- Class name: `IFRNet VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The IFRNet_VFI node is designed for video frame interpolation, leveraging deep learning models to predict and generate intermediate frames between existing frames in a video sequence. This node aims to enhance video fluidity and quality by filling in missing frames, making it suitable for applications in video editing, slow-motion effects, and improving video streaming experiences.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name specifies the pre-trained model to be used for frame interpolation. It is crucial for determining the interpolation model's parameters and behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - This input represents the sequence of frames to be interpolated. It is essential for providing the raw data from which intermediate frames will be generated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Specifies after how many frames the cache should be cleared to prevent memory overflow. It is important for managing memory usage during the interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of original frames. It is key to determining the smoothness of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_factor`**
    - The scale factor adjusts the resolution of the interpolated frames. It is significant for matching the output frame size with the original frames or upscaling/downscaling as needed.
    - Comfy dtype: `COMBO[FLOAT]`
    - Python dtype: `float`
### Optional
- **`optional_interpolation_states`**
    - Optional states that can skip certain frames during interpolation, useful for customizing the interpolation process based on specific requirements.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - Determines whether the cache should be stored in fp16 format to save memory. It is important for optimizing memory usage during the interpolation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - This output consists of the interpolated frames generated between the initial and final frames of the video sequence, enhancing video fluidity and quality.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IFRNet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
                "cache_in_fp16": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        clear_cache_after_n_frames: typing.SupportsInt = 1,
        multiplier: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        optional_interpolation_states: InterpolationStateList = None,
        cache_in_fp16: bool = True
    ):
        from .IFRNet_S_arch import IRFNet_S
        from .IFRNet_L_arch import IRFNet_L
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IRFNet_S() if 'S' in ckpt_name else IRFNet_L()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor):
            return model(frame_0, frame_1, timestep, scale_factor)
        
        args = [interpolation_model, scale_factor]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        )
        return (out,)

```
