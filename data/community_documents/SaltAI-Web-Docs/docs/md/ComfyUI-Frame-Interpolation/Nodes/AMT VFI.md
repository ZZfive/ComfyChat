# AMT VFI
## Documentation
- Class name: `AMT VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The AMT_VFI node is designed to facilitate advanced video frame interpolation by leveraging the AMT (Adaptive Motion Transfer) architecture. It aims to enhance video quality by generating intermediate frames that result in smoother transitions and improved visual fidelity.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the AMT model to be loaded, directly influencing the interpolation quality and performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - The sequence of frames to be interpolated. This input is essential for determining the starting and ending points of the interpolation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Controls the frequency of cache clearing to manage memory usage during the interpolation process, affecting performance and efficiency.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of input frames, directly impacting the smoothness of the output video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - Provides optional states for skipping specific frames during interpolation, allowing for customized frame processing.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - Determines whether to cache frames in half-precision format to save memory, influencing the trade-off between memory usage and precision.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a sequence of interpolated frames, showcasing the node's ability to enhance video smoothness and visual quality through advanced frame interpolation techniques.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AMT_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_CONFIGS.keys()), ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 1, "min": 1, "max": 100}),
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
        model_path = load_file_from_direct_url(MODEL_TYPE, f"https://huggingface.co/lalala125/AMT/resolve/main/{ckpt_name}")
        ckpt_config = CKPT_CONFIGS[ckpt_name]

        interpolation_model = ckpt_config["network"](**ckpt_config["params"])
        interpolation_model.load_state_dict(torch.load(model_path)["state_dict"])
        interpolation_model.eval().to(get_torch_device())

        frames = preprocess_frames(frames)
        padder = InputPadder(frames.shape, 16)
        frames = padder.pad(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            return model(
                frame_0, 
                frame_1,
                embt=torch.FloatTensor([timestep] * frame_0.shape[0]).view(frame_0.shape[0], 1, 1, 1).to(get_torch_device()),
                scale_factor=1.0,
                eval=True
            )["imgt_pred"]
        
        args = [interpolation_model]
        out = generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        out = padder.unpad(out)
        out = postprocess_frames(out)
        return (out,)

```
