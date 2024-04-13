# ControlNet Soft Weights 🛂🅐🅒🅝
## Documentation
- Class name: `SoftControlNetWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet`
- Output node: `False`

This node is designed for loading and applying soft control net weights with customizable base multipliers and optional locking mechanisms. It enables dynamic adjustment of control net weights based on a mask, facilitating fine-grained control over the weight application process.
## Input types
### Required
- **`weight_00`**
    - Specifies the first weight in the sequence, allowing for granular control over the initial weight value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_01`**
    - Defines the second weight in the sequence, contributing to the nuanced adjustment of control net weights.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_02`**
    - Represents the third weight in the sequence, enabling further customization of the weight distribution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_03`**
    - Indicates the fourth weight in the sequence, allowing for detailed control over the weight application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_04`**
    - Specifies the fifth weight in the sequence, contributing to the overall weight configuration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_05`**
    - Defines the sixth weight in the sequence, enhancing the flexibility of weight adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_06`**
    - Represents the seventh weight in the sequence, enabling precise control over the weight distribution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_07`**
    - Indicates the eighth weight in the sequence, allowing for detailed customization of the weight application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_08`**
    - Specifies the ninth weight in the sequence, contributing to the overall weight configuration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_09`**
    - Defines the tenth weight in the sequence, enhancing the flexibility of weight adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_10`**
    - Represents the eleventh weight in the sequence, enabling precise control over the weight distribution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_11`**
    - Indicates the twelfth weight in the sequence, allowing for detailed customization of the weight application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_12`**
    - Specifies the thirteenth weight in the sequence, contributing to the overall weight configuration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean parameter that determines whether the weights should be flipped, offering an additional layer of control.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Represents the computed control net weights after applying the mask and base multipliers.
    - Python dtype: `ControlWeights`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A timestep keyframe object that encapsulates the control weights for use in further processing or visualization.
    - Python dtype: `TimestepKeyframe`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.09941396206337118, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.12050177219802567, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.14606275417942507, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 0.17704576264172736, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 0.214600924414215, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 0.26012233262329093, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 0.3152997971191405, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 0.3821815722656249, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 0.4632503906249999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 0.561515625, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 0.6806249999999999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
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
