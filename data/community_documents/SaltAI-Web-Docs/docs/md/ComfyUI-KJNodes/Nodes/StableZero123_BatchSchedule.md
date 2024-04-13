# StableZero123_BatchSchedule
## Documentation
- Class name: `StableZero123_BatchSchedule`
- Category: `KJNodes`
- Output node: `False`

The `StableZero123_BatchSchedule` node is designed to manage and optimize the scheduling of batch processing tasks. It focuses on efficiently organizing the execution of tasks that are part of a batch, aiming to enhance throughput and reduce processing time by leveraging advanced scheduling techniques.
## Input types
### Required
- **`clip_vision`**
    - This input is essential for providing visual context or features that are crucial for the batch scheduling process, influencing how tasks are prioritized and arranged.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.Tensor`
- **`init_image`**
    - Represents the initial image data that may be used as a reference or starting point in the batch processing tasks, affecting the scheduling based on visual content requirements.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - A variational autoencoder model input that could be used for tasks involving image generation or manipulation within the batch, influencing the scheduling based on computational needs.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`width`**
    - Specifies the width dimension for images involved in the batch tasks, impacting the scheduling by defining the size constraints of processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Defines the height dimension for images involved in the batch tasks, similarly impacting the scheduling by setting size constraints for processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Determines the number of tasks or items to be processed in a single batch, directly affecting the efficiency and organization of the batch scheduling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - Specifies the interpolation method to be used in tasks that require image resizing, influencing the scheduling based on the computational complexity of different methods.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`azimuth_points_string`**
    - A string representing key points for azimuth adjustments in 3D model processing tasks, affecting the scheduling by defining specific requirements for task execution.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`elevation_points_string`**
    - A string detailing key points for elevation adjustments in 3D model tasks, impacting the scheduling by outlining specific execution requirements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The output representing the positively conditioned data or results from the batch processing tasks, organized for efficient execution.
    - Python dtype: `torch.Tensor`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - This output includes negatively conditioned data or results, providing a contrast or alternative outcomes from the batch processing tasks.
    - Python dtype: `torch.Tensor`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Represents the latent representations or features extracted from the batch processing tasks, essential for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StableZero123_BatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
                              "azimuth_points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                              "elevation_points_string": ("STRING", {"default": "0:(0.0),\n7:(0.0),\n15:(0.0)\n", "multiline": True}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "KJNodes"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, azimuth_points_string, elevation_points_string, interpolation):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        def ease_in(t):
            return t * t
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)
        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the azimuth input string into a list of tuples
        azimuth_points = []
        azimuth_points_string = azimuth_points_string.rstrip(',\n')
        for point_str in azimuth_points_string.split(','):
            frame_str, azimuth_str = point_str.split(':')
            frame = int(frame_str.strip())
            azimuth = float(azimuth_str.strip()[1:-1]) 
            azimuth_points.append((frame, azimuth))
        # Sort the points by frame number
        azimuth_points.sort(key=lambda x: x[0])

        # Parse the elevation input string into a list of tuples
        elevation_points = []
        elevation_points_string = elevation_points_string.rstrip(',\n')
        for point_str in elevation_points_string.split(','):
            frame_str, elevation_str = point_str.split(':')
            frame = int(frame_str.strip())
            elevation_val = float(elevation_str.strip()[1:-1]) 
            elevation_points.append((frame, elevation_val))
        # Sort the points by frame number
        elevation_points.sort(key=lambda x: x[0])

        # Index of the next point to interpolate towards
        next_point = 1
        next_elevation_point = 1

        positive_cond_out = []
        positive_pooled_out = []
        negative_cond_out = []
        negative_pooled_out = []
        
        #azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            # If next_point is equal to the length of points, we've gone past the last point
            if next_point == len(azimuth_points):
                next_point -= 1  # Set next_point to the last index of points
            prev_point = max(next_point - 1, 0)  # Ensure prev_point is not less than 0

            # Calculate fraction
            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:  # Prevent division by zero
                fraction = (i - azimuth_points[prev_point][0]) / (azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                # Use the new interpolate_angle function
                interpolated_azimuth = interpolate_angle(azimuth_points[prev_point][1], azimuth_points[next_point][1], fraction)
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]
            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                fraction = (i - elevation_points[prev_elevation_point][0]) / (elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                if interpolation == "ease_in":
                    fraction = ease_in(fraction)
                elif interpolation == "ease_out":
                    fraction = ease_out(fraction)
                elif interpolation == "ease_in_out":
                    fraction = ease_in_out(fraction)
                
                interpolated_elevation = interpolate_angle(elevation_points[prev_elevation_point][1], elevation_points[next_elevation_point][1], fraction)
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            cam_embeds = camera_embeddings(interpolated_elevation, interpolated_azimuth)
            cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

            positive_pooled_out.append(t)
            positive_cond_out.append(cond)
            negative_pooled_out.append(torch.zeros_like(t))
            negative_cond_out.append(torch.zeros_like(pooled))

        # Concatenate the conditions and pooled outputs
        final_positive_cond = torch.cat(positive_cond_out, dim=0)
        final_positive_pooled = torch.cat(positive_pooled_out, dim=0)
        final_negative_cond = torch.cat(negative_cond_out, dim=0)
        final_negative_pooled = torch.cat(negative_pooled_out, dim=0)

        # Structure the final output
        final_positive = [[final_positive_cond, {"concat_latent_image": final_positive_pooled}]]
        final_negative = [[final_negative_cond, {"concat_latent_image": final_negative_pooled}]]

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (final_positive, final_negative, {"samples": latent})

```
