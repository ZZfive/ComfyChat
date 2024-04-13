# 🔧 Image Crop
## Documentation
- Class name: `ImageCrop+`
- Category: `essentials`
- Output node: `False`

The ImageCrop+ node is designed for cropping images to a specified width and height from a given starting point (x, y). This operation allows for the selective focus on certain parts of an image, which can be crucial for tasks that require attention to specific image regions or for aesthetic adjustments.
## Input types
### Required
- **`image`**
    - The image to be cropped. This parameter is crucial as it defines the source image that will undergo the cropping process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the width of the cropped area. It determines how wide the resulting image will be after cropping.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the cropped area. It determines how tall the resulting image will be after cropping.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`position`**
    - Specifies the position of the cropping area relative to the original image, which can be adjusted further with offsets.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x_offset`**
    - The horizontal offset to be applied to the cropping area, allowing for fine-tuned positioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y_offset`**
    - The vertical offset to be applied to the cropping area, allowing for fine-tuned positioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The cropped image. This output is the result of the cropping operation, containing only the specified area of the original image.
    - Python dtype: `torch.Tensor`
- **`x`**
    - Comfy dtype: `INT`
    - The x-coordinate of the top-left corner of the cropped area in the original image.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The y-coordinate of the top-left corner of the cropped area in the original image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ImageSharpen](../../Comfy/Nodes/ImageSharpen.md)
    - SetNode
    - [IPAdapterApply](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApply.md)
    - Reroute
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [MiDaS-DepthMapPreprocessor](../../comfyui_controlnet_aux/Nodes/MiDaS-DepthMapPreprocessor.md)
    - [IPAdapterApplyFaceID](../../ComfyUI_IPAdapter_plus/Nodes/IPAdapterApplyFaceID.md)



## Source code
```python
class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","x","y",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)
                
        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width
        
        x += x_offset
        y += y_offset
        
        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, x, y, )

```
