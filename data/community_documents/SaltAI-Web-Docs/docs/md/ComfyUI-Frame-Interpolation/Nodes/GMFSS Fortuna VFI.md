# GMFSS Fortuna VFI
## Documentation
- Class name: `GMFSS Fortuna VFI`
- Category: `ComfyUI-Frame-Interpolation/VFI`
- Output node: `False`

The GMFSS Fortuna VFI node is designed for video frame interpolation, leveraging deep learning models to predict intermediate frames between two given frames. It utilizes a specialized architecture to enhance the quality and accuracy of the interpolated frames, aiming to achieve seamless transitions in video sequences.
## Input types
### Required
- **`ckpt_name`**
    - The name of the checkpoint file for the model. It is crucial for loading the specific model configuration and weights for frame interpolation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - A tensor containing the sequence of frames for interpolation. It serves as the input data from which intermediate frames will be generated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clear_cache_after_n_frames`**
    - Specifies after how many frames the CUDA cache should be cleared to prevent memory overflow. It helps in managing memory usage during the interpolation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - Defines the number of intermediate frames to be generated between each pair of input frames. It determines the smoothness of the output video sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_interpolation_states`**
    - An optional parameter that allows for skipping certain frames during interpolation, based on predefined states. It can be used to optimize the interpolation process.
    - Comfy dtype: `INTERPOLATION_STATES`
    - Python dtype: `InterpolationStateList`
- **`cache_in_fp16`**
    - A boolean flag indicating whether to cache frames in half-precision format (fp16) to save memory. It affects the memory efficiency of the interpolation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output tensor containing the interpolated frames, showcasing the model's ability to create high-quality, seamless transitions between the input frames.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)



## Source code
```python
class GMFSS_Fortuna_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPTS_PATH_CONFIG.keys()), ),
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
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        cache_in_fp16: bool = True
    ):
        """
        Perform video frame interpolation using a given checkpoint model.
    
        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
    
        Returns:
            tuple: A tuple containing the output interpolated frames.
    
        Note:
            This method interpolates frames in a video sequence using a specified checkpoint model. 
            It processes each frame sequentially, generating interpolated frames between them.
    
            To prevent memory overflow, it clears the CUDA cache after processing a specified number of frames.
        """
        
        interpolation_model = CommonModelInference(model_type=ckpt_name)
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep, model, scale):
            return model(frame_0, frame_1, timestep, scale)
        
        scale = 1
        
        args = [interpolation_model, scale]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float16 if cache_in_fp16 else torch.float32)
        )
        return (out,)

```
