# T2IAdapter Soft Weights 🛂🅐🅒🅝
## Documentation
- Class name: `SoftT2IAdapterWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter`
- Output node: `False`

The SoftT2IAdapterWeights node is designed to adjust the influence of control weights on the generation process in a soft, adaptable manner. It allows for fine-tuning the balance between different aspects of generated content by modifying the weights applied to control nets, enhancing the flexibility and precision of content generation.
## Input types
### Required
- **`weight_00`**
    - Specifies the initial weight for the first control net, allowing for fine-tuning of its influence on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_01`**
    - Defines the weight for the second control net, enabling adjustments to its impact on content generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_02`**
    - Sets the weight for the third control net, facilitating control over its contribution to the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_03`**
    - Determines the weight for the fourth control net, allowing for precise manipulation of its effect on the generation outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that, when set to True, reverses the order of weights, potentially altering the generation dynamics.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The adjusted control net weights after applying the specified modifications.
    - Python dtype: `List[float]`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A timestep keyframe object that encapsulates the adjusted control weights for use in the generation process.
    - Python dtype: `TimestepKeyframe`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SoftT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
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
