# Paste By Mask
## Documentation
- Class name: `Paste By Mask`
- Category: `Masquerade Nodes`
- Output node: `False`

The node is designed to overlay one image onto another based on a specified mask, with options to adjust the resizing behavior of the pasted image to ensure it fits within the mask area appropriately. It supports different resizing strategies to maintain the aspect ratio or match the source image dimensions, providing flexibility in how the image is integrated into the base image.
## Input types
### Required
- **`image_base`**
    - The base image onto which another image will be pasted. It serves as the background for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_to_paste`**
    - The image to be overlaid onto the base image. This image is manipulated according to the mask and resizing behavior to fit appropriately.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A mask that defines the area on the base image where the second image will be pasted. It plays a crucial role in determining the overlay's boundaries.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`resize_behavior`**
    - Specifies how the image to be pasted should be resized to fit within the mask area. It affects the final appearance of the pasted image on the base image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`mask_mapping_optional`**
    - An optional mapping to adjust how the mask is applied, allowing for more complex pasting scenarios beyond a straightforward overlay.
    - Comfy dtype: `MASK_MAPPING`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of the pasting operation, which is the base image with the second image overlaid according to the mask and resizing behavior.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)



## Source code
```python
class PasteByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("IMAGE",),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],)
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"

    CATEGORY = "Masquerade Nodes"

    def paste(self, image_base, image_to_paste, mask, resize_behavior, mask_mapping_optional = None):
        image_base = tensor2rgba(image_base)
        image_to_paste = tensor2rgba(image_to_paste)
        mask = tensor2mask(mask)

        # Scale the mask to be a matching size if it isn't
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]
        if mask_mapping_optional is None:
            if B < PB:
                assert(PB % B == 0)
                image_base = image_base.repeat(PB // B, 1, 1, 1)
            B, H, W, C = image_base.shape
            if MB < B:
                assert(B % MB == 0)
                mask = mask.repeat(B // MB, 1, 1)
            elif B < MB:
                assert(MB % B == 0)
                image_base = image_base.repeat(MB // B, 1, 1, 1)
            if PB < B:
                assert(B % PB == 0)
                image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, MH, MW = mask.shape

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        for i in range(0, MB):
            if is_empty[i]:
                continue
            else:
                image_index = i
                if mask_mapping_optional is not None:
                    image_index = mask_mapping_optional[i].item()
                source_size = image_to_paste.size()
                SB, SH, SW, _ = image_to_paste.shape

                # Figure out the desired size
                width = int(target_width[i].item())
                height = int(target_height[i].item())
                if resize_behavior == "keep_ratio_fill":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        width = int(height * actual_ratio)
                    elif actual_ratio < target_ratio:
                        height = int(width / actual_ratio)
                elif resize_behavior == "keep_ratio_fit":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        height = int(width / actual_ratio)
                    elif actual_ratio < target_ratio:
                        width = int(height * actual_ratio)
                elif resize_behavior == "source_size" or resize_behavior == "source_size_unmasked":
                    width = SW
                    height = SH

                # Resize the image we're pasting if needed
                resized_image = image_to_paste[i].unsqueeze(0)
                if SH != height or SW != width:
                    resized_image = torch.nn.functional.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

                pasting = torch.ones([H, W, C])
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0

                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= (xmax - W)
                    xmax = W
                if ymax > H:
                    source_ymax -= (ymax - H)
                    ymax = H

                pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
                pasting[:, :, 3] = 1.

                pasting_alpha = torch.zeros([H, W])
                pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

                if resize_behavior == "keep_ratio_fill" or resize_behavior == "source_size_unmasked":
                    # If we explicitly want to fill the area, we are ok with extending outside
                    paste_mask = pasting_alpha.unsqueeze(2).repeat(1, 1, 4)
                else:
                    paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)
                result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)
        return (result,)

```
