# BatchCropFromMask
## Documentation
- Class name: `BatchCropFromMask`
- Category: `KJNodes/masking`
- Output node: `False`

The BatchCropFromMask node is designed for processing a batch of images and their corresponding masks to crop them based on the maximum bounding box size calculated across all masks. It dynamically adjusts the crop size to ensure optimal coverage and uniformity across the batch, incorporating smoothing and scaling techniques to refine the bounding box dimensions.
## Input types
### Required
- **`original_images`**
    - The original_images parameter consists of the images that correspond to the input masks. These images are subject to cropping based on the calculated bounding boxes derived from the masks.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`masks`**
    - The masks parameter represents the input masks for the images to be cropped. It plays a crucial role in determining the areas of interest within each image, guiding the cropping process.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
- **`crop_size_mult`**
    - The crop_size_mult parameter is a multiplier that scales the size of the calculated bounding box. It allows for fine-tuning the crop size, accommodating different use cases and preferences.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_smooth_alpha`**
    - The bbox_smooth_alpha parameter is used to smooth the changes in bounding box size across the batch of images and masks. This smoothing helps in achieving more consistent crop sizes and shapes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`original_images`**
    - Comfy dtype: `IMAGE`
    - The original images that were input for cropping, returned without modification.
    - Python dtype: `List[torch.Tensor]`
- **`cropped_images`**
    - Comfy dtype: `IMAGE`
    - The images after being cropped according to the calculated bounding boxes.
    - Python dtype: `List[torch.Tensor]`
- **`bboxes`**
    - Comfy dtype: `BBOX`
    - The bounding boxes calculated for each image/mask pair.
    - Python dtype: `List[Tuple[int, int, int, int]]`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the bounding box used for cropping.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the bounding box used for cropping.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchCropFromMask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_size_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "bbox_smooth_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "BBOX",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "original_images",
        "cropped_images",
        "bboxes",
        "width",
        "height",
    )
    FUNCTION = "crop"
    CATEGORY = "KJNodes/masking"

    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if alpha == 0:
            return prev_bbox_size
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if alpha == 0:
            return prev_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1])
        )

    def crop(self, masks, original_images, crop_size_mult, bbox_smooth_alpha):
 
        bounding_boxes = []
        cropped_images = []

        self.max_bbox_width = 0
        self.max_bbox_height = 0

        # First, calculate the maximum bounding box size across all masks
        curr_max_bbox_width = 0
        curr_max_bbox_height = 0
        for mask in masks:
            _mask = tensor2pil(mask)[0]
            non_zero_indices = np.nonzero(np.array(_mask))
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            width = max_x - min_x
            height = max_y - min_y
            curr_max_bbox_width = max(curr_max_bbox_width, width)
            curr_max_bbox_height = max(curr_max_bbox_height, height)

        # Smooth the changes in the bounding box size
        self.max_bbox_width = self.smooth_bbox_size(self.max_bbox_width, curr_max_bbox_width, bbox_smooth_alpha)
        self.max_bbox_height = self.smooth_bbox_size(self.max_bbox_height, curr_max_bbox_height, bbox_smooth_alpha)

        # Apply the crop size multiplier
        self.max_bbox_width = round(self.max_bbox_width * crop_size_mult)
        self.max_bbox_height = round(self.max_bbox_height * crop_size_mult)
        bbox_aspect_ratio = self.max_bbox_width / self.max_bbox_height

        # # Make sure max_bbox_size is divisible by 32, if not, round it upwards so it is
        # self.max_bbox_width = math.ceil(self.max_bbox_width / 32) * 32
        # self.max_bbox_height = math.ceil(self.max_bbox_height / 32) * 32

        # Then, for each mask and corresponding image...
        for i, (mask, img) in enumerate(zip(masks, original_images)):
            _mask = tensor2pil(mask)[0]
            non_zero_indices = np.nonzero(np.array(_mask))
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            
            # Calculate center of bounding box
            center_x = np.mean(non_zero_indices[1])
            center_y = np.mean(non_zero_indices[0])
            curr_center = (round(center_x), round(center_y))

            # If this is the first frame, initialize prev_center with curr_center
            if not hasattr(self, 'prev_center'):
                self.prev_center = curr_center

            # Smooth the changes in the center coordinates from the second frame onwards
            if i > 0:
                center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
            else:
                center = curr_center

            # Update prev_center for the next frame
            self.prev_center = center

            # Create bounding box using max_bbox_width and max_bbox_height
            half_box_width = round(self.max_bbox_width / 2)
            half_box_height = round(self.max_bbox_height / 2)
            min_x = max(0, center[0] - half_box_width)
            max_x = min(img.shape[1], center[0] + half_box_width)
            min_y = max(0, center[1] - half_box_height)
            max_y = min(img.shape[0], center[1] + half_box_height)

            # Append bounding box coordinates
            bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

            # Crop the image from the bounding box
            cropped_img = img[min_y:max_y, min_x:max_x, :]
            
            # Calculate the new dimensions while maintaining the aspect ratio
            new_height = min(cropped_img.shape[0], self.max_bbox_height)
            new_width = round(new_height * bbox_aspect_ratio)

            # Resize the image
            resize_transform = Resize((new_height, new_width))
            resized_img = resize_transform(cropped_img.permute(2, 0, 1))

            # Perform the center crop to the desired size
            crop_transform = CenterCrop((self.max_bbox_height, self.max_bbox_width)) # swap the order here if necessary
            cropped_resized_img = crop_transform(resized_img)

            cropped_images.append(cropped_resized_img.permute(1, 2, 0))

        cropped_out = torch.stack(cropped_images, dim=0)
        
        return (original_images, cropped_out, bounding_boxes, self.max_bbox_width, self.max_bbox_height, )

```
