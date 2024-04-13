# CreateVoronoiMask
## Documentation
- Class name: `CreateVoronoiMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

This node is designed to generate Voronoi diagrams as masks. It leverages the mathematical properties of Voronoi tessellation to partition the space based on a set of seed points, creating visually distinct regions that can be used as masks for various image processing and generative art applications.
## Input types
### Required
- **`frames`**
    - Specifies the number of frames for the Voronoi mask animation, affecting the temporal length of the generated mask sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`num_points`**
    - Determines the number of seed points used in the Voronoi tessellation, directly influencing the complexity and visual density of the resulting mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`line_width`**
    - Controls the thickness of the lines between Voronoi cells, allowing for adjustment of the visual prominence of the cell boundaries.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`speed`**
    - Adjusts the speed at which the Voronoi cells evolve over time in the mask animation, offering control over the dynamic visual effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`frame_width`**
    - Sets the width of the frame for the Voronoi mask, defining the horizontal dimension of the output mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - Sets the height of the frame for the Voronoi mask, defining the vertical dimension of the output mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The primary output mask generated from the Voronoi tessellation, usable for various image processing applications.
    - Python dtype: `torch.Tensor`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - An inverted version of the primary Voronoi mask, offering an alternative visual representation for creative applications.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateVoronoiMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createvoronoi"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "frames": ("INT", {"default": 16,"min": 2, "max": 4096, "step": 1}),
                 "num_points": ("INT", {"default": 15,"min": 1, "max": 4096, "step": 1}),
                 "line_width": ("INT", {"default": 4,"min": 1, "max": 4096, "step": 1}),
                 "speed": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 1.0, "step": 0.01}),
                 "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                 "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createvoronoi(self, frames, num_points, line_width, speed, frame_width, frame_height):
        # Define the number of images in the batch
        batch_size = frames
        out = []
          
        # Calculate aspect ratio
        aspect_ratio = frame_width / frame_height
        
        # Create start and end points for each point, considering the aspect ratio
        start_points = np.random.rand(num_points, 2)
        start_points[:, 0] *= aspect_ratio
        
        end_points = np.random.rand(num_points, 2)
        end_points[:, 0] *= aspect_ratio

        for i in range(batch_size):
            # Interpolate the points' positions based on the current frame
            t = (i * speed) / (batch_size - 1)  # normalize to [0, 1] over the frames
            t = np.clip(t, 0, 1)  # ensure t is in [0, 1]
            points = (1 - t) * start_points + t * end_points  # lerp

            # Adjust points for aspect ratio
            points[:, 0] *= aspect_ratio

            vor = Voronoi(points)

            # Create a blank image with a white background
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xlim([0, aspect_ratio]); ax.set_ylim([0, 1])  # adjust x limits
            ax.axis('off')
            ax.margins(0, 0)
            fig.set_size_inches(aspect_ratio * frame_height/100, frame_height/100)  # adjust figure size
            ax.fill_between([0, 1], [0, 1], color='white')

            # Plot each Voronoi ridge
            for simplex in vor.ridge_vertices:
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', linewidth=line_width)

            fig.canvas.draw()
            img = np.array(fig.canvas.renderer._renderer)

            plt.close(fig)

            pil_img = Image.fromarray(img).convert("L")
            mask = torch.tensor(np.array(pil_img)) / 255.0

            out.append(mask)

        return (torch.stack(out, dim=0), 1.0 - torch.stack(out, dim=0),)

```
