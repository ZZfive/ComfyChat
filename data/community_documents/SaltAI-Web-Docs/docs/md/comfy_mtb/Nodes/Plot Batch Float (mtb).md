# Plot Batch Float (mtb)
## Documentation
- Class name: `Plot Batch Float (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The Plot Batch Float node is designed for visualizing float values over time, creating a plot that can be customized in terms of dimensions and appearance. It allows for the dynamic representation of data, facilitating the analysis and interpretation of trends and patterns.
## Input types
### Required
- **`width`**
    - Specifies the width of the plot. A larger width allows for a more detailed visualization of the data.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the plot, affecting the vertical scale and potentially the readability of the plot.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`point_size`**
    - Controls the size of the points plotted, which can enhance clarity or focus on specific data points.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - Used to initialize the random number generator, ensuring reproducibility of the plot's appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`plot`**
    - Comfy dtype: `IMAGE`
    - The generated plot as an image, encapsulating the visualized float values over time.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PlotBatchFloat:
    """Plot floats"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768}),
                "height": ("INT", {"default": 768}),
                "point_size": ("INT", {"default": 4}),
                "seed": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "plot"
    CATEGORY = "mtb/batch"

    def plot(self, width, height, point_size, seed, **kwargs):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.set_edgecolor("black")
        fig.patch.set_facecolor("#2e2e2e")
        # Setting background color and grid
        ax.set_facecolor("#2e2e2e")  # Dark gray background
        ax.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # Finding global min and max across all lists for scaling the plot
        global_min = min(min(values) for values in kwargs.values())
        global_max = max(max(values) for values in kwargs.values())

        # Color cycle to ensure each plot has a distinct color
        colormap = plt.cm.get_cmap("viridis", len(kwargs))
        color_normalization_factor = (
            0.5 if len(kwargs) == 1 else (len(kwargs) - 1)
        )

        # Plotting each list with a unique color
        for i, (label, values) in enumerate(kwargs.items()):
            color_value = i / color_normalization_factor
            ax.plot(values, label=label, color=colormap(color_value))

        ax.set_ylim(global_min, global_max)  # Scaling the y-axis
        ax.legend(
            title="Legend",
            title_fontsize="large",
            fontsize="medium",
            edgecolor="black",
        )

        # Setting labels and title
        ax.set_xlabel("Time", fontsize="large", color="white")
        ax.set_ylabel("Value", fontsize="large", color="white")
        ax.set_title(
            "Plot of Values over Time", fontsize="x-large", color="white"
        )

        # Adjusting tick colors to be visible on dark background
        ax.tick_params(colors="white")

        # Changing color of the axes border
        for _, spine in ax.spines.items():
            spine.set_edgecolor("white")

        # Rendering the plot into a NumPy array
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf)
        plt.close(fig)  # Closing the figure to free up memory

        return (pil2tensor(image),)

    def draw_point(self, image, point, color, point_size):
        x, y = point
        y = image.shape[0] - 1 - y  # Invert Y-coordinate
        half_size = point_size // 2
        x_start, x_end = (
            max(0, x - half_size),
            min(image.shape[1], x + half_size + 1),
        )
        y_start, y_end = (
            max(0, y - half_size),
            min(image.shape[0], y + half_size + 1),
        )
        image[y_start:y_end, x_start:x_end] = color

    def draw_line(self, image, start, end, color):
        x1, y1 = start
        x2, y2 = end

        # Invert Y-coordinate
        y1 = image.shape[0] - 1 - y1
        y2 = image.shape[0] - 1 - y2

        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1
        dy = y2 - y1
        error = int(dx / 2.0)
        y = y1
        ystep = None
        if y1 < y2:
            ystep = 1
        else:
            ystep = -1
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            image[coord] = color
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        if swapped:
            image[(x1, y1)] = color
            image[(x2, y2)] = color

```
