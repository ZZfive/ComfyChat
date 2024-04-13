# T2IAdapter Custom Weights 🛂🅐🅒🅝
## Documentation
- Class name: `CustomT2IAdapterWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter`
- Output node: `False`

This node is designed for loading and customizing weights for the T2I (Text-to-Image) adapter within the Advanced ControlNet framework. It allows for the adjustment of individual weight parameters and the option to flip the weights, facilitating fine-tuned control over the image generation process.
## Input types
### Required
- **`weight_00`**
    - Specifies the initial weight value, allowing for fine-tuning of the first weight parameter in the T2I adapter's control mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_01`**
    - Defines the second weight parameter, enabling precise adjustments to influence the image generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_02`**
    - Sets the third weight parameter, contributing to the detailed customization of the T2I adapter's behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_03`**
    - Determines the fourth weight parameter, offering further control over the image synthesis process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that, when set to True, reverses the order of weights, potentially altering the image generation outcome.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The customized control net weights after applying the specified adjustments.
    - Python dtype: `list[float]`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A timestep keyframe object that encapsulates the control weights for use in the image generation process.
    - Python dtype: `TimestepKeyframe`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CustomT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03]
        weights = get_properly_arranged_t2i_weights(weights)
        weights = ControlWeights.t2iadapter(weights, flip_weights=flip_weights)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
