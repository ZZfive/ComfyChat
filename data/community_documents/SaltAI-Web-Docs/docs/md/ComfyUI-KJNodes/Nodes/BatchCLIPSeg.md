# BatchCLIPSeg
## Documentation
- Class name: `BatchCLIPSeg`
- Category: `KJNodes/masking`
- Output node: `False`

The BatchCLIPSeg node is designed to perform image segmentation tasks by leveraging the capabilities of the CLIPSeg model. It abstracts the complexity of processing and segmenting images using the CLIPSeg architecture, providing an efficient and streamlined way to obtain segmented images from input data.
## Input types
### Required
- **`images`**
    - The 'images' input is essential for the segmentation process, as it provides the raw images that the BatchCLIPSeg node will process using the CLIPSeg model. The quality and characteristics of these images directly influence the segmentation outcome.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`text`**
    - The 'text' input allows users to specify textual descriptions or queries that guide the segmentation process. This input is crucial for tailoring the segmentation results to match specific concepts or objects described in the text, thereby affecting the node's execution and results.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`threshold`**
    - The 'threshold' input determines the sensitivity of the segmentation process. A higher threshold may result in more precise segmentation but could also miss finer details, while a lower threshold might capture more details but also include more noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`binary_mask`**
    - The 'binary_mask' input provides an optional mask that can be used to focus the segmentation process on specific areas of the input images. This can significantly affect the segmentation results by limiting the areas that are considered for segmentation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `torch.Tensor`
- **`combine_mask`**
    - The 'combine_mask' input, if provided, allows for the combination of the segmentation results with a pre-existing mask. This can be used to refine or adjust the segmentation outcome based on additional criteria or masks.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `torch.Tensor`
- **`use_cuda`**
    - The 'use_cuda' input specifies whether to use CUDA for processing. Enabling CUDA can significantly speed up the segmentation process by leveraging GPU acceleration, affecting the node's performance and efficiency.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`Mask`**
    - Comfy dtype: `MASK`
    - This output represents the segmented mask produced by the BatchCLIPSeg node. The mask highlights the areas of the input images that match the textual descriptions provided, offering a visual representation of the segmentation results.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchCLIPSeg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":
                    {
                        "images": ("IMAGE",),
                        "text": ("STRING", {"multiline": False}),
                        "threshold": ("FLOAT", {"default": 0.15,"min": 0.0, "max": 10.0, "step": 0.01}),
                        "binary_mask": ("BOOLEAN", {"default": True}),
                        "combine_mask": ("BOOLEAN", {"default": False}),
                        "use_cuda": ("BOOLEAN", {"default": True}),
                     },
                }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)

    FUNCTION = "segment_image"

    def segment_image(self, images, text, threshold, binary_mask, combine_mask, use_cuda):        

        out = []
        height, width, _ = images[0].shape
        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        model.to(device)  # Ensure the model is on the correct device
        images = images.to(device)
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        for image in images:
            image = (image* 255).type(torch.uint8)
            prompt = text
            input_prc = processor(text=prompt, images=image, padding="max_length", return_tensors="pt")
            # Move the processed input to the device
            for key in input_prc:
                input_prc[key] = input_prc[key].to(device)
            
            outputs = model(**input_prc)
            tensor = torch.sigmoid(outputs[0])
        
            tensor_thresholded = torch.where(tensor > threshold, tensor, torch.tensor(0, dtype=torch.float))
            tensor_normalized = (tensor_thresholded - tensor_thresholded.min()) / (tensor_thresholded.max() - tensor_thresholded.min())

            tensor = tensor_normalized

            # Add extra dimensions to the mask for batch and channel
            tensor = tensor[None, None, :, :]

            # Resize the mask
            resized_tensor = F.interpolate(tensor, size=(height, width), mode='bilinear', align_corners=False)

            # Remove the extra dimensions
            resized_tensor = resized_tensor[0, 0, :, :]

            out.append(resized_tensor)
          
        results = torch.stack(out).cpu()
        
        if combine_mask:
            combined_results = torch.max(results, dim=0)[0]
            results = combined_results.unsqueeze(0).repeat(len(images),1,1)

        if binary_mask:
            results = results.round()
        
        return results,

```
