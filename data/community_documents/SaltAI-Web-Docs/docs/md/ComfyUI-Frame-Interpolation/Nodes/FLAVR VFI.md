# FLAVR VFI
## Documentation
- Class name: `FLAVR VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

FLAVR VFI is designed for frame interpolation, specifically focusing on 2x interpolation. It utilizes a deep learning model to enhance video quality by interpolating frames, aiming to create smoother transitions and improve visual fluidity.
## Input types
### Required
- **`ckpt_name`**
    - The name of the checkpoint file to load the pre-trained FLAVR model, crucial for initializing the model with learned weights for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - The input video frames to be interpolated. These frames are processed to generate intermediate frames, enhancing the video's smoothness.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Specifies the number of frames processed before clearing the CUDA cache to avoid memory overflow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Specifies the interpolation factor. Currently, FLAVR only supports 2x interpolation, which is essential for achieving the desired frame enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`duplicate_first_last_frames`**
    - Indicates whether to duplicate the first and last frames to potentially enhance the visual continuity of the interpolated video.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_interpolation_states`**
    - Optional states that can be used to influence the interpolation process, providing flexibility in handling different video processing scenarios.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `Dict`
- **`cache_in_fp16`**
    - Determines if the cache should be stored in fp16 format to potentially reduce memory usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The interpolated frames produced by the FLAVR model, resulting in a smoother and visually enhanced video.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FLAVR_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 2}), #TODO: Implement recursively invoking interpolator for multi-frame interpolation
                "duplicate_first_last_frames": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
                "cache_in_fp16": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"        

    #Reference: https://github.com/danier97/ST-MFNet/blob/main/interpolate_yuv.py#L93
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        duplicate_first_last_frames: bool = False,
        optional_interpolation_states: InterpolationStateList = None,
        cache_in_fp16: bool = True
    ):
        if multiplier != 2:
            warnings.warn("Currently, FLAVR only supports 2x interpolation. The process will continue but please set multiplier=2 afterward")

        assert_batch_size(frames, batch_size=4, vfi_name="ST-MFNet")
        interpolation_states = optional_interpolation_states
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        model = build_flavr(model_path)
        frames = preprocess_frames(frames)
        padder = InputPadder(frames.shape, 16)
        frames = padder.pad(frames)

        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        for frame_itr in range(len(frames) - 3):
            #Does skipping frame i+1 make sanse in this case?
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr) and interpolation_states.is_frame_skipped(frame_itr + 1):
                continue
            
            #Ensure that input frames are in fp32 - the same dtype as model
            frame0, frame1, frame2, frame3 = (
                frames[frame_itr:frame_itr+1].float(),
                frames[frame_itr+1:frame_itr+2].float(), 
                frames[frame_itr+2:frame_itr+3].float(), 
                frames[frame_itr+3:frame_itr+4].float()
            )
            new_frame = model([frame0.to(device), frame1.to(device), frame2.to(device), frame3.to(device)])[0].detach().cpu()
            number_of_frames_processed_since_last_cleared_cuda_cache += 2
            
            if frame_itr == 0:
                output_frames.append(frame0)
                if duplicate_first_last_frames:
                    output_frames.append(frame0) # repeat the first frame
                output_frames.append(frame1)
            output_frames.append(new_frame)
            output_frames.append(frame2)
            if frame_itr == len(frames) - 4:
                output_frames.append(frame3)
                if duplicate_first_last_frames:
                    output_frames.append(frame3) # repeat the last frame

            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end = ' ')
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")
            gc.collect()
        
        dtype = torch.float16 if cache_in_fp16 else torch.float32
        output_frames = [frame.cpu().to(dtype=dtype) for frame in output_frames] #Ensure all frames are in cpu
        out = torch.cat(output_frames, dim=0)
        out = padder.unpad(out)
        # clear cache for courtesy
        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )

```
