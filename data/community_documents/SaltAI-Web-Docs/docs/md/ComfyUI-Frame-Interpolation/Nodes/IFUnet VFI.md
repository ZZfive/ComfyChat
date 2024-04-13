# IFUnet VFI
## Documentation
- Class name: `IFUnet VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The IFUnet_VFI node is designed for video frame interpolation, leveraging deep learning models to predict and generate intermediate frames between existing frames in a video sequence. This process enhances video fluidity and can be used for various applications such as slow-motion video generation, video restoration, and improving video frame rates.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the IFUnet model, determining the specific pre-trained model weights to be used for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`frames`**
    - The sequence of frames to be interpolated. This input is crucial for determining the start and end points of the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`clear_cache_after_n_frames`**
    - Controls the cache clearing mechanism to manage memory usage during the interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of original frames, directly affecting the smoothness of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_factor`**
    - Determines the scaling factor for the output frames, allowing for adjustments in the resolution of the interpolated frames.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ensemble`**
    - A boolean flag that enables or disables the ensemble method for frame interpolation, potentially improving the quality of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_interpolation_states`**
    - Provides optional states for frame interpolation, allowing for more control over which frames are interpolated.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - Determines whether the cache should be stored in fp16 format, optimizing memory usage during the interpolation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a sequence of interpolated frames, enhancing the fluidity of the original video by filling in gaps between frames.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IFUnet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
                "ensemble": ("BOOLEAN", {"default":True})
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
        ensemble: bool = True,
        optional_interpolation_states: InterpolationStateList = None,
        cache_in_fp16: bool = True
    ):
        from .IFUNet_arch import IFUNetModel
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IFUNetModel()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor, ensemble):
            return model(frame_0, frame_1, timestep=timestep, scale=scale_factor, ensemble=ensemble)
        
        args = [interpolation_model, scale_factor, ensemble]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        )
        return (out,)

```
