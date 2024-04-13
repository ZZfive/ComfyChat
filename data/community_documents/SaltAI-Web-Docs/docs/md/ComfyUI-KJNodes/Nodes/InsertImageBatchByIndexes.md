# InsertImageBatchByIndexes
## Documentation
- Class name: `InsertImageBatchByIndexes`
- Category: `KJNodes`
- Output node: `False`

This node is designed for inserting a batch of images into another batch of images at specified indexes, maintaining the original sequence order. It is particularly useful in scenarios where selective augmentation or modification of an image batch is required.
## Input types
### Required
- **`images`**
    - The original batch of images into which the new images will be inserted. It serves as the base for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`images_to_insert`**
    - The batch of images to be inserted into the original batch at specified indexes. This parameter allows for selective augmentation or modification of the image batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`insert_indexes`**
    - A list of indexes specifying where in the original batch each image from 'images_to_insert' should be placed. It dictates the insertion points within the original batch.
    - Comfy dtype: `INDEXES`
    - Python dtype: `List[int]`
## Output types
- **`images_after_insert`**
    - Comfy dtype: `IMAGE`
    - The updated batch of images after the insertion operation, maintaining the original sequence order.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InsertImageBatchByIndexes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), 
                "images_to_insert": ("IMAGE",), 
                "insert_indexes": ("INDEXES",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images_after_insert",
    )
    FUNCTION = "insert"
    CATEGORY = "KJNodes"
    
    def insert(self, images, images_to_insert, insert_indexes):
        """
        This node is designed to be use with node FilterZeroMasksAndCorrespondingImages
        It inserts the images_to_insert into images according to insert_indexes

        Returns:
            images_after_insert: updated original images with origonal sequence order 
        """
        
        images_after_insert = images
        
        if images_to_insert is not None and insert_indexes is not None:
            images_to_insert_num = len(images_to_insert)
            insert_indexes_num = len(insert_indexes)
            if images_to_insert_num == insert_indexes_num:
                images_after_insert = []

                i_images = 0
                for i in range(len(images) + images_to_insert_num):
                    if i in insert_indexes:
                        images_after_insert.append(images_to_insert[insert_indexes.index(i)])
                    else:
                        images_after_insert.append(images[i_images])
                        i_images += 1
                        
                images_after_insert = torch.stack(images_after_insert, dim=0)
                
            else:
                print(f"[WARNING] skip this node, due to number of images_to_insert ({images_to_insert_num}) is not equal to number of insert_indexes ({insert_indexes_num})")


        return (images_after_insert, )

```
