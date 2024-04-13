# Image Overlay
## Documentation
- Class name: `Image Overlay`
- Category: `List Stuff`
- Output node: `False`

The ImageLabelOverlay node is designed to overlay textual labels on images. It supports the addition of labels as floating-point numbers, integers, or strings, allowing for versatile labeling options across a variety of image types.
## Input types
### Required
- **`images`**
    - A list of images on which labels are to be overlaid. This parameter is essential for defining the base images that will be modified by the overlay process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
### Optional
- **`float_labels`**
    - Optional floating-point labels to overlay on the images. These labels provide a way to annotate images with numerical data for visualization or identification purposes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Optional[List[float]]`
- **`int_labels`**
    - Optional integer labels to overlay on the images. Similar to float_labels, these provide a means to annotate images with numerical identifiers or categories.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[List[int]]`
- **`str_labels`**
    - Optional string labels to overlay on the images. This allows for textual annotations, offering a flexible way to label images with names, descriptions, or any textual information.
    - Comfy dtype: `STR`
    - Python dtype: `Optional[List[str]]`
## Output types
- **`Images`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageLabelOverlay:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "float_labels": ("FLOAT", {"forceInput": True}),
                "int_labels": ("INT", {"forceInput": True}),
                "str_labels": ("STR", {"forceInput": True}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "put_overlay"

    CATEGORY = "List Stuff"

    def put_overlay(
            self,
            images: List[Tensor],
            float_labels: Optional[List[float]] = None,
            int_labels: Optional[List[int]] = None,
            str_labels: Optional[List[str]] = None,
    ) -> Tuple[List[Tensor]]:
        batches = images

        labels_to_check: Dict[str, Union[List[float], List[int], List[str], None]] = {
            "float": float_labels if float_labels is not None else None,
            "int": int_labels if int_labels is not None else None,
            "str": str_labels if str_labels is not None else None
        }

        for l_type, labels in labels_to_check.items():
            if labels is None:
                continue
            if len(batches) != len(labels) and len(labels) != 1:
                raise Exception(
                    f"Non-matching input sizes got {len(batches)} Image Batches, {len(labels)} Labels for label type {l_type}"
                )

        image_h, _, _ = batches[0][0].size()

        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), 60)

        ret_images: List[Tensor]= []
        loop_gen = zip_with_fill(batches, float_labels, int_labels, str_labels)
        for b_idx, (img_batch, float_lbl, int_lbl, str_lbl) in enumerate(loop_gen):
            batch: List[Tensor] = []
            for i_idx, img in enumerate(img_batch):
                pil_img = tensor2pil(img)
                # print(f"Batch: {b_idx} | img: {i_idx}")
                # print(img.size())
                draw = ImageDraw.Draw(pil_img)

                draw.text((0, image_h - 60), f"B: {b_idx} | I: {i_idx}", fill="red", font=font)

                y_offset = 0
                for _, lbl in zip(["float", "int", "str"], [float_lbl, int_lbl, str_lbl]):
                    if lbl is None:
                        continue
                    draw.rectangle((0, 0 + y_offset, 512, 60 + y_offset), fill="#ffff33")
                    draw.text((0, 0 + y_offset), str(lbl), fill="red", font=font)
                    y_offset += 60
                batch.append(pil2tensor(pil_img))

            ret_images.append(torch.cat(batch))

        return (ret_images,)

```
