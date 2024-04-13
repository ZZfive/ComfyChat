# ImageTransformByNormalizedAmplitude
## Documentation
- Class name: `ImageTransformByNormalizedAmplitude`
- Category: `KJNodes`
- Output node: `False`

This node is designed to transform images based on normalized amplitude values. It adjusts the zoom level of images incrementally or cumulatively, depending on the amplitude, to create a series of transformed images. The transformation process involves cropping and resizing operations to simulate zoom effects, tailored by the amplitude's influence.
## Input types
### Required
- **`normalized_amp`**
    - An array of normalized amplitude values, each corresponding to an image in the batch. These values dictate the extent of zoom applied to each image, influencing the transformation process.
    - Comfy dtype: `NORMALIZED_AMPLITUDE`
    - Python dtype: `numpy.ndarray`
- **`zoom_scale`**
    - A scaling factor that determines the intensity of the zoom effect applied to the images. It modulates the transformation based on the amplitude values.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cumulative`**
    - A boolean flag indicating whether the zoom effect should be applied cumulatively across the images. When set to true, it enhances the transformation effect progressively.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`image`**
    - The input image or batch of images to be transformed. It serves as the base for applying zoom transformations based on amplitude values.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A list of images that have been transformed based on the provided amplitude values and zoom scale. Each image in the list represents a zoomed version of the original, adjusted according to its corresponding amplitude.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageTransformByNormalizedAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "normalized_amp": ("NORMALIZED_AMPLITUDE",),
            "zoom_scale": ("FLOAT", { "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "display": "number" }),
            "cumulative": ("BOOLEAN", { "default": False }),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "amptransform"
    CATEGORY = "KJNodes"

    def amptransform(self, image, normalized_amp, zoom_scale, cumulative):
        # Ensure normalized_amp is an array and within the range [0, 1]
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        transformed_images = []

        # Initialize the cumulative zoom factor
        prev_amp = 0.0

        for i in range(image.shape[0]):
            img = image[i]  # Get the i-th image in the batch
            amp = normalized_amp[i]  # Get the corresponding amplitude value

            # Incrementally increase the cumulative zoom factor
            if cumulative:
                prev_amp += amp
                amp += prev_amp

            # Convert the image tensor from BxHxWxC to CxHxW format expected by torchvision
            img = img.permute(2, 0, 1)
            
            # Convert PyTorch tensor to PIL Image for processing
            pil_img = TF.to_pil_image(img)
            
            # Calculate the crop size based on the amplitude
            width, height = pil_img.size
            crop_size = int(min(width, height) * (1 - amp * zoom_scale))
            crop_size = max(crop_size, 1)
            
            # Calculate the crop box coordinates (centered crop)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = (width + crop_size) // 2
            bottom = (height + crop_size) // 2
            
            # Crop and resize back to original size
            cropped_img = TF.crop(pil_img, top, left, crop_size, crop_size)
            resized_img = TF.resize(cropped_img, (height, width))
            
            # Convert back to tensor in CxHxW format
            tensor_img = TF.to_tensor(resized_img)
            
            # Convert the tensor back to BxHxWxC format
            tensor_img = tensor_img.permute(1, 2, 0)
            
            # Add to the list
            transformed_images.append(tensor_img)
        
        # Stack all transformed images into a batch
        transformed_batch = torch.stack(transformed_images)
        
        return (transformed_batch,)

```
