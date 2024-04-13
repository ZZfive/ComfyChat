# comfyui-upscale-by-model

This custom node allow upscaling an image by a factor using a model.

![screenshot](screenshot.png)

## Usage

This node will do the following steps:

- Upscale the input image with the upscale model.
- Check the size of the upscaled image.
    - If the upscaled size is larger than the target size (calculated from the upscale factor `upscale_by`), then downscale the image to the target size using the scaling method defined by `rescale_method`.
    - If the upscaled size is smaller than or equal to the target size, then do nothing.
- Return the upscaled image.

### Input

- `upscale_model`: Take an upscale model.
- `image`: The image to upscale.
- `upscale_by`: The factor to upscale the image by.
- `rescale_method`: The method to downscale the image to the target scale.

### Output

- `IMAGE`: The upscaled image.

## Install

Clone this repository into ComfyUI's `custom_nodes` folder.