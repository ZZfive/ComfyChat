# 🔧 Image Resize
## Documentation
- Class name: `ImageResize+`
- Category: `essentials`
- Output node: `False`

The ImageResize+ node is designed for resizing images to specified dimensions, offering a range of methods to achieve the desired scaling while potentially preserving the aspect ratio or applying different interpolation techniques.
## Input types
### Required
- **`image`**
    - The input image to be resized.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - The target width for the resized image. If not specified, it may be calculated based on the height and the original aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The target height for the resized image. If not specified, it may be calculated based on the width and the original aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - The method used for scaling the image, which can affect the quality and characteristics of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`keep_proportion`**
    - Indicates whether to preserve the original aspect ratio of the image during resizing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`condition`**
    - A conditional input that may affect the resizing process based on specific criteria or settings.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The resized image.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The actual width of the resized image.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The actual height of the resized image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - SetNode
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - [PrepImageForClipVision](../../ComfyUI_IPAdapter_plus/Nodes/PrepImageForClipVision.md)



## Source code
```python
class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "condition": (["always", "only if bigger", "only if smaller"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, keep_proportion, interpolation="nearest", condition="always"):
        if keep_proportion is True:
            _, oh, ow, _ = image.shape

            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            #width = ow if width == 0 else width
            #height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)

        outputs = p(image)

        if "always" in condition or ("bigger" in condition and (oh > height or ow > width)) or ("smaller" in condition and (oh < height or ow < width)):
            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)
        
        outputs = pb(outputs)

        return(outputs, outputs.shape[2], outputs.shape[1],)

```
