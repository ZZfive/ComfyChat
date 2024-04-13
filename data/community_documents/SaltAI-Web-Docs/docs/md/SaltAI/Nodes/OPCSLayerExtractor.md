# Parallax Motion Camera Scheduler Extractor
## Documentation
- Class name: `OPCSLayerExtractor`
- Category: `SALT/Animation/Keyframing/Parallax Motion`
- Output node: `False`

The OPCSLayerExtractor node is designed for extracting specific animation layer data from a collection of animation frames. It focuses on isolating zoom, x, and y values for a given layer, facilitating the analysis or manipulation of these parameters within the context of parallax motion animation.
## Input types
### Required
- **`float_layers`**
    - A list of animation frames, where each frame is represented as a list of layer data. This input is essential for selecting and extracting the desired layer's data.
    - Comfy dtype: `LIST`
    - Python dtype: `List[List[float]]`
- **`layer_index`**
    - The index of the layer to be extracted from the animation frames. This allows for targeted extraction of data from a specific layer, enhancing the node's flexibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`zoom_values`**
    - Comfy dtype: `LIST`
    - A list of zoom values extracted from the specified layer across all frames.
    - Python dtype: `List[float]`
- **`x_values`**
    - Comfy dtype: `LIST`
    - A list of x-coordinate values extracted from the specified layer across all frames.
    - Python dtype: `List[float]`
- **`y_values`**
    - Comfy dtype: `LIST`
    - A list of y-coordinate values extracted from the specified layer across all frames.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPCSLayerExtractor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_layers": ("LIST", ),
                "layer_index": ("INT", {"default": 0, "min": 0})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom_values", "x_values", "y_values")
    FUNCTION = "extract"
    CATEGORY = "SALT/Animation/Keyframing/Parallax Motion"

    def extract(self, **kwargs):
        animation_data = kwargs.get("float_layers", [])
        layer_index = kwargs.get("layer_index", 0)

        if layer_index >= len(animation_data):
            raise ValueError("Layer index out of range.")

        selected_layer_data = animation_data[layer_index]
        zoom_values = [frame[0] for frame in selected_layer_data]
        x_values = [frame[1] for frame in selected_layer_data]
        y_values = [frame[2] for frame in selected_layer_data]

        print("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("Zoom Values", zoom_values)
        log_curve("X Values", x_values)
        log_curve("Y Values", y_values)

        return zoom_values, x_values, y_values

```
