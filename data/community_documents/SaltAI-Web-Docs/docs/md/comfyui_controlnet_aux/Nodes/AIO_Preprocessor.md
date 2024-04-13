# AIO Aux Preprocessor
## Documentation
- Class name: `AIO_Preprocessor`
- Category: `ControlNet Preprocessors`
- Output node: `False`

The AIO_Preprocessor node serves as a dynamic image preprocessing unit within the ControlNet Preprocessors category, capable of selecting and executing a variety of auxiliary preprocessing functions based on the specified preprocessor type. This flexibility allows for tailored image preprocessing to enhance control net performance.
## Input types
### Required
- **`image`**
    - The input image to be preprocessed. It serves as the primary data upon which the selected preprocessing operation is applied, directly influencing the output and effectiveness of the preprocessing step.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image.Image`
### Optional
- **`preprocessor`**
    - Specifies the auxiliary preprocessing function to be used for image preprocessing. This selection determines the specific preprocessing operation applied to the input image, thereby influencing the preprocessing outcome and directly affecting the control net's ability to interpret and process the image effectively.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`resolution`**
    - Defines the resolution at which the image should be processed. This parameter can significantly impact the preprocessing outcome by affecting the level of detail preserved or altered during the operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image output, which has undergone the specified auxiliary preprocessing operation to potentially enhance its suitability for control net applications.
    - Python dtype: `PIL.Image.Image`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ControlNetApply](../../Comfy/Nodes/ControlNetApply.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)



## Source code
```python
class AIO_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        auxs = list(AUX_NODE_MAPPINGS.keys())
        for name in AIO_NOT_SUPPORTED:
            if name in auxs: auxs.remove(name)
        
        return create_node_input_types(
            preprocessor=(auxs, {"default": "CannyEdgePreprocessor"})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, preprocessor, image, resolution=512):
        aux_class = AUX_NODE_MAPPINGS[preprocessor]
        input_types = aux_class.INPUT_TYPES()
        input_types = {
            **input_types["required"], 
            **(input_types["optional"] if "optional" in input_types else {})
        }
        params = {}
        for name, input_type in input_types.items():
            if name == "image":
                params[name] = image
                continue
            
            if name == "resolution":
                params[name] = resolution
                continue
            
            if len(input_type) == 2 and ("default" in input_type[1]):
                params[name] = input_type[1]["default"]
                continue

            default_values = { "INT": 0, "FLOAT": 0.0 }
            if input_type[0] in default_values:
                params[name] = default_values[input_type[0]]

        return getattr(aux_class(), aux_class.FUNCTION)(**params)

```
