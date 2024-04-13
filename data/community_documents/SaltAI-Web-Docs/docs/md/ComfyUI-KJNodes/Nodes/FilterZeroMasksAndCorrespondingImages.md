# FilterZeroMasksAndCorrespondingImages
## Documentation
- Class name: `FilterZeroMasksAndCorrespondingImages`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to filter out all zero-value masks from a given set of masks and, if provided, filter out the corresponding images based on the indexes of the zero-value masks. It ensures that the output contains only masks with non-zero values and their corresponding images, maintaining data integrity and relevance for further processing.
## Input types
### Required
- **`masks`**
    - The 'masks' parameter represents the collection of masks to be filtered. Masks with all zero values are removed, ensuring that only masks with non-zero values are processed, which is crucial for maintaining data relevance and integrity in image processing tasks.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
### Optional
- **`original_images`**
    - The 'original_images' parameter is optional and represents the collection of images corresponding to the masks. If provided, images corresponding to zero-value masks are filtered out, maintaining alignment between the filtered masks and images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[List[torch.Tensor]]`
## Output types
- **`non_zero_masks_out`**
    - Comfy dtype: `MASK`
    - The filtered non-zero masks, ensuring that only masks with non-zero values are processed for further steps.
    - Python dtype: `torch.Tensor`
- **`non_zero_mask_images_out`**
    - Comfy dtype: `IMAGE`
    - The corresponding images for the non-zero masks, if original images were provided.
    - Python dtype: `Optional[torch.Tensor]`
- **`zero_mask_images_out`**
    - Comfy dtype: `IMAGE`
    - The images corresponding to zero-value masks, useful for analysis or further processing.
    - Python dtype: `Optional[torch.Tensor]`
- **`zero_mask_images_out_indexes`**
    - Comfy dtype: `INDEXES`
    - The indexes of the zero-value masks, useful for tracking and analysis purposes.
    - Python dtype: `Optional[List[int]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FilterZeroMasksAndCorrespondingImages:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "original_images": ("IMAGE",), 
            },
        }

    RETURN_TYPES = (
        "MASK",
        "IMAGE",
        "IMAGE",
        "INDEXES"
    )
    RETURN_NAMES = (
        "non_zero_masks_out",
        "non_zero_mask_images_out",
        "zero_mask_images_out",
        "zero_mask_images_out_indexes"
    )
    FUNCTION = "filter"
    CATEGORY = "KJNodes/masking"
    
    def filter(self, masks, original_images=None):
        """
        Filter out all the empty (i.e. all zero) mask in masks
        Also filter out all the corresponding images in original_images by indexes if provide

        Args:
            original_images (optional): If provide, it need have same length as masks.
        """
        non_zero_masks = []
        non_zero_mask_images = []
        zero_mask_images = []
        zero_mask_images_indexes = []
        
        masks_num = len(masks)
        also_process_images = False
        if original_images is not None:
            imgs_num = len(original_images)
            if len(original_images) == masks_num:
                also_process_images = True
            else:
                print(f"[WARNING] ignore input: original_images, due to number of original_images ({imgs_num}) is not equal to number of masks ({masks_num})")
        
        for i in range(masks_num):
            non_zero_num = np.count_nonzero(np.array(masks[i]))
            if non_zero_num > 0:
                non_zero_masks.append(masks[i])
                if also_process_images:
                    non_zero_mask_images.append(original_images[i])
            else:
                zero_mask_images.append(original_images[i])
                zero_mask_images_indexes.append(i)

        non_zero_masks_out = torch.stack(non_zero_masks, dim=0)
        non_zero_mask_images_out = zero_mask_images_out = zero_mask_images_out_indexes = None
        
        if also_process_images:
            non_zero_mask_images_out = torch.stack(non_zero_mask_images, dim=0)
            if len(zero_mask_images) > 0:
                zero_mask_images_out = torch.stack(zero_mask_images, dim=0)
                zero_mask_images_out_indexes = zero_mask_images_indexes

        return (non_zero_masks_out, non_zero_mask_images_out, zero_mask_images_out, zero_mask_images_out_indexes)

```
