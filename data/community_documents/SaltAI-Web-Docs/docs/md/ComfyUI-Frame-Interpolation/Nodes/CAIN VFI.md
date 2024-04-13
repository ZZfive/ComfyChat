# CAIN VFI
## Documentation
- Class name: `CAIN VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The CAIN_VFI node encapsulates the functionality of the CAIN model for video frame interpolation, leveraging deep learning techniques to predict intermediate frames between two consecutive frames in a video sequence. It aims to enhance video fluidity and generate high-quality intermediate frames by understanding and processing the temporal and spatial information contained within the video frames.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name specifies the pre-trained model to be used for frame interpolation, playing a crucial role in determining the quality and effectiveness of the generated frames.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - The input frames tensor contains the consecutive frames between which the intermediate frames are to be interpolated. This tensor is critical for the model to analyze and understand the motion and content present in the video sequence.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - This parameter controls the frequency of cache clearing during frame interpolation, affecting memory management and potentially the speed of the interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - The multiplier determines the number of intermediate frames to be generated between each pair of input frames, directly influencing the smoothness and fluidity of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Optional states for interpolation that can be used to customize or optimize the interpolation process, providing flexibility in handling different video content or requirements.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - This flag indicates whether the cache should be stored in half-precision floating-point format (FP16), which can reduce memory usage at the cost of potential precision loss.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a tensor containing the interpolated frames, enhancing the original video sequence with additional frames to improve smoothness and visual quality.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CAIN_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000})
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
        from .cain_arch import CAIN
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        sd = torch.load(model_path)["state_dict"]
        sd = {key.replace('module.', ''): value for key, value in sd.items()}


        global interpolation_model
        interpolation_model = CAIN(depth=3)
        interpolation_model.load_state_dict(sd)
        interpolation_model.eval().to(get_torch_device())
        del sd

        frames = preprocess_frames(frames)
    
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            #CAIN does some direct modifications to input frame tensors so we need to clone them
            return model(frame_0.detach().clone(), frame_1.detach().clone())[0]
        
        args = [interpolation_model]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, use_timestep=False, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        )
        return (out,)

```
