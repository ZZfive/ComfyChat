# ControlNet Custom Weights 🛂🅐🅒🅝
## Documentation
- Class name: `CustomControlNetWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet`
- Output node: `False`

This node is designed to load and process custom control net weights, applying specific weight values across a range of parameters and optionally flipping the weights. It's part of the advanced control network system, enabling precise manipulation of the generation process through detailed control weight adjustments.
## Input types
### Required
- **`weight_00`**
    - Specifies the weight for the first control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_01`**
    - Specifies the weight for the second control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_02`**
    - Specifies the weight for the third control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_03`**
    - Specifies the weight for the fourth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_04`**
    - Specifies the weight for the fifth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_05`**
    - Specifies the weight for the sixth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_06`**
    - Specifies the weight for the seventh control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_07`**
    - Specifies the weight for the eighth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_08`**
    - Specifies the weight for the ninth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_09`**
    - Specifies the weight for the tenth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_10`**
    - Specifies the weight for the eleventh control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_11`**
    - Specifies the weight for the twelfth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_12`**
    - Specifies the weight for the thirteenth control point, influencing the generation process by adjusting its impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that, when set to True, reverses the order of the control weights, potentially altering the generation outcome.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The processed control net weights after applying the specified weights and optionally flipping, ready for use in the generation process.
    - Python dtype: `list[float]`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A shortcut to a timestep keyframe object that encapsulates the control weights, facilitating their integration into the generation pipeline.
    - Python dtype: `TimestepKeyframe`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CustomControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        weights = ControlWeights.controlnet(weights, flip_weights=flip_weights)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
