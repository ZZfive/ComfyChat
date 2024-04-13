# M2M VFI
## Documentation
- Class name: `M2M VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The M2M_VFI node is designed for frame interpolation in video sequences, leveraging deep learning models to predict intermediate frames between existing ones, enhancing video fluidity and frame rate.
## Input types
### Required
- **`ckpt_name`**
    - The name of the checkpoint to load the model from. It determines the specific model configuration and weights used for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - A tensor containing the frames of the video sequence. It is the primary input from which intermediate frames are generated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Specifies after how many frames the cache should be cleared to manage memory usage during processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Determines the number of intermediate frames to be generated between each pair of input frames, affecting the smoothness of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Optional states that can be passed to control the interpolation process, allowing for customization of the frame generation.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - A flag indicating whether to cache data in half-precision format to reduce memory consumption.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The interpolated frames generated between the input frames, enhancing the video's smoothness and frame rate.
    - Python dtype: `List[torch.Tensor]`
- **`ui`**
    - A parameter representing the user interface elements associated with the node's output, facilitating interaction and visualization.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class M2M_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
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
        optional_interpolation_states: InterpolationStateList = None,
        cache_in_fp16: bool = True
    ):
        from .M2M_arch import M2M_PWC
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = M2M_PWC()
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, int_timestep, model):
            tenSteps = [
                torch.FloatTensor([int_timestep] * len(frame_0)).view(len(frame_0), 1, 1, 1).to(get_torch_device())
            ]
            return model(frame_0, frame_1, tenSteps)[0]
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        )
        return (out,)

```
