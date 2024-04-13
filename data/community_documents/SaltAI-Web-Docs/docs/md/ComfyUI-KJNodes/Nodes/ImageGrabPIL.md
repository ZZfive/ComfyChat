# ImageGrabPIL
## Documentation
- Class name: `ImageGrabPIL`
- Category: `KJNodes/experimental`
- Output node: `False`

The ImageGrabPIL node is designed for capturing and processing images directly from the screen using the PIL library. It focuses on grabbing specific regions of the screen, allowing for the extraction of images without the need for pre-existing image files. This node is particularly useful for applications requiring real-time image capture and processing.
## Input types
### Required
- **`x`**
    - Specifies the starting x-coordinate of the screen region to capture.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - Specifies the starting y-coordinate of the screen region to capture.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Defines the width of the screen region to be captured.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Defines the height of the screen region to be captured.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`num_frames`**
    - Determines the number of frames to capture in a sequence, allowing for the creation of image sequences or simple animations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`delay`**
    - Sets the delay between captures when multiple frames are being captured, controlling the speed of the sequence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single image or a sequence of images captured from the specified screen region, ready for further processing or analysis.
    - Python dtype: `PIL.Image or List[PIL.Image]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)
    - [CannyEdgePreprocessor](../../comfyui_controlnet_aux/Nodes/CannyEdgePreprocessor.md)
    - [PixelPerfectResolution](../../comfyui_controlnet_aux/Nodes/PixelPerfectResolution.md)



## Source code
```python
class ImageGrabPIL:

    @classmethod
    def IS_CHANGED(cls):

        return

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "screencap"
    CATEGORY = "KJNodes/experimental"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
                 "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
                 "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
        },
    } 

    def screencap(self, x, y, width, height, num_frames, delay):
        captures = []
        bbox = (x, y, x + width, y + height)
        
        for _ in range(num_frames):
            # Capture screen
            screen_capture = ImageGrab.grab(bbox=bbox)
            screen_capture_torch = torch.tensor(np.array(screen_capture), dtype=torch.float32) / 255.0
            screen_capture_torch = screen_capture_torch.unsqueeze(0)
            captures.append(screen_capture_torch)
            
            # Wait for a short delay if more than one frame is to be captured
            if num_frames > 1:
                time.sleep(delay)
        
        return (torch.cat(captures, dim=0),)

```
