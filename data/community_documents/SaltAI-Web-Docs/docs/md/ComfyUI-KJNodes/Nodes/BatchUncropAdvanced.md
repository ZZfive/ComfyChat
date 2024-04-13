# BatchUncropAdvanced
## Documentation
- Class name: `BatchUncropAdvanced`
- Category: `KJNodes/masking`
- Output node: `False`

The BatchUncropAdvanced node is designed for advanced operations of uncropping images in a batch. It integrates additional parameters and functionalities to handle complex uncropping scenarios, including the use of combined masks and bounding boxes, to precisely restore the original dimensions and content of cropped images.
## Input types
### Required
- **`original_images`**
    - Original images are the base images before any cropping operation was applied. They serve as a reference for the uncropping process to restore the images to their original state.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`cropped_images`**
    - Cropped images are the result of a previous cropping operation. This node aims to reverse that operation and restore these images to their original dimensions and content.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`cropped_masks`**
    - Cropped masks indicate the areas of the original images that were retained during the cropping process. They are essential for accurately reconstructing the original images.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
- **`combined_crop_mask`**
    - The combined crop mask is a single mask that represents the cumulative area covered by all individual cropped masks. It is used when a unified approach to uncropping is preferred.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`bboxes`**
    - Bounding boxes define the coordinates of the cropped areas within the original images. They are crucial for positioning the cropped content correctly during the uncropping process.
    - Comfy dtype: `BBOX`
    - Python dtype: `List[Tuple[int, int, int, int]]`
- **`border_blending`**
    - Border blending specifies the method used to blend the borders of the uncropped areas with the surrounding content, ensuring a seamless transition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`crop_rescale`**
    - Crop rescale determines whether and how the cropped images should be rescaled before being placed back into their original context.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_combined_mask`**
    - This flag indicates whether the combined crop mask should be used instead of individual cropped masks for the uncropping operation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_square_mask`**
    - This flag determines whether a square mask should be applied to the cropped areas, potentially altering the shape of the area to be uncropped.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`combined_bounding_box`**
    - The combined bounding box is an optional parameter that defines a single bounding box encompassing all cropped areas. It is used for operations where a unified bounding box is more effective than individual ones.
    - Comfy dtype: `BBOX`
    - Python dtype: `Optional[Tuple[int, int, int, int]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a list of images that have been uncropped, with their original dimensions and content restored as closely as possible.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchUncropAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",), 
                "cropped_masks": ("MASK",),
                "combined_crop_mask": ("MASK",),
                "bboxes": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_combined_mask": ("BOOLEAN", {"default": False}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "combined_bounding_box": ("BBOX", {"default": None}),  
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "uncrop"

    CATEGORY = "KJNodes/masking"

    def uncrop(self, original_images, cropped_images, cropped_masks, combined_crop_mask, bboxes, border_blending, crop_rescale, use_combined_mask, use_square_mask, combined_bounding_box = None):
        
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
            return bordered_image

        if len(original_images) != len(cropped_images) or len(original_images) != len(bboxes):
            raise ValueError("The number of images, crop_images, and bboxes should be the same")

        crop_imgs = tensor2pil(cropped_images)
        input_images = tensor2pil(original_images)
        out_images = []

        for i in range(len(input_images)):
            img = input_images[i]
            crop = crop_imgs[i]
            bbox = bboxes[i]
            
            if use_combined_mask:
                bb_x, bb_y, bb_width, bb_height = combined_bounding_box[0]
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = combined_crop_mask[i]
            else:
                bb_x, bb_y, bb_width, bb_height = bbox
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = cropped_masks[i]
            
            # scale paste_region
            scale_x = scale_y = crop_rescale
            paste_region = (round(paste_region[0]*scale_x), round(paste_region[1]*scale_y), round(paste_region[2]*scale_x), round(paste_region[3]*scale_y))

            # rescale the crop image to fit the paste_region
            crop = crop.resize((round(paste_region[2]-paste_region[0]), round(paste_region[3]-paste_region[1])))
            crop_img = crop.convert("RGB")

            #border blending
            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
            blend = img.convert("RGBA")

            if use_square_mask:
                mask = Image.new("L", img.size, 0)
                mask_block = Image.new("L", (paste_region[2]-paste_region[0], paste_region[3]-paste_region[1]), 255)
                mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
                mask.paste(mask_block, paste_region)
            else:
                original_mask = tensor2pil(mask)[0]
                original_mask = original_mask.resize((paste_region[2]-paste_region[0], paste_region[3]-paste_region[1]))
                mask = Image.new("L", img.size, 0)
                mask.paste(original_mask, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            blend.paste(crop_img, paste_region) 
            blend.putalpha(mask)
            
            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)

```
