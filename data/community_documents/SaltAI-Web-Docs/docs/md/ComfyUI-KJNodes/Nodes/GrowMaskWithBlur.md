# GrowMaskWithBlur
## Documentation
- Class name: `GrowMaskWithBlur`
- Category: `KJNodes/masking`
- Output node: `False`

The GrowMaskWithBlur node is designed to modify the size and shape of input masks through expansion or contraction, while optionally applying a blur effect to soften the edges. This node combines the functionalities of mask manipulation with image processing techniques to achieve more nuanced and visually appealing results.
## Input types
### Required
- **`mask`**
    - The mask input represents the target area to be modified. Its expansion or contraction is determined by other parameters, affecting the final output's shape and size.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`expand`**
    - This parameter specifies the degree of expansion or contraction applied to the mask. Positive values cause the mask to grow, while negative values lead to its shrinkage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`incremental_expandrate`**
    - Defines the rate at which the mask's expansion or contraction occurs incrementally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tapered_corners`**
    - Determines whether the corners of the expanded or contracted mask should be tapered, adding a more natural transition at the edges.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`flip_input`**
    - Indicates whether the input mask should be flipped horizontally or vertically before processing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blur_radius`**
    - Specifies the radius of the blur effect to be applied, controlling the softness of the mask's edges after modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`lerp_alpha`**
    - Specifies the alpha value for linear interpolation, affecting the blending of the mask during the expansion or contraction process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`decay_factor`**
    - Determines the rate at which the effect of expansion, contraction, or blur decays over time or distance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a modified version of the input mask, potentially expanded or contracted and optionally blurred, depending on the provided parameters.
    - Python dtype: `torch.Tensor`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - An additional output providing the inverted version of the modified mask, offering an alternative perspective on the mask manipulation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GrowMaskWithBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "incremental_expandrate": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "flip_input": ("BOOLEAN", {"default": False}),
                "blur_radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100,
                    "step": 0.1
                }),
                "lerp_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    CATEGORY = "KJNodes/masking"

    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "expand_mask"
    
    def expand_mask(self, mask, expand, tapered_corners, flip_input, blur_radius, incremental_expandrate, lerp_alpha, decay_factor):
        alpha = lerp_alpha
        decay = decay_factor
        if( flip_input ):
            mask = 1.0 - mask
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        previous_output = None
        for m in growmask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if expand < 0:
                expand -= abs(incremental_expandrate)  # Use abs(growrate) to ensure positive change
            else:
                expand += abs(incremental_expandrate)  # Use abs(growrate) to ensure positive change
            output = torch.from_numpy(output)
            if alpha < 1.0 and previous_output is not None:
                # Interpolate between the previous and current frame
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                # Add the decayed previous output to the current frame
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output)

        if blur_radius != 0:
            # Convert the tensor list to PIL images, apply blur, and convert back
            for idx, tensor in enumerate(out):
                # Convert tensor to PIL image
                pil_image = tensor2pil(tensor.cpu().detach())[0]
                # Apply Gaussian blur
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                # Convert back to tensor
                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)
            return (blurred, 1.0 - blurred)
        else:
            return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)

```
