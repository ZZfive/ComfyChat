# Keyframing Convert to FLOATS (MTB)
## Documentation
- Class name: `OPAC2Floats`
- Category: `SALT/Animation/Keyframing`
- Output node: `False`

The OPAC2Floats node is designed to convert lists of animation parameters into float values, facilitating the transition from list-based inputs to a format suitable for numerical operations and further processing in animation keyframing.
## Input types
### Required
### Optional
- **`zoom`**
    - Converts a list of zoom values into float values, enabling precise control over the zoom aspect of animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`angle`**
    - Transforms a list of angle values into float values, allowing for detailed adjustments to the angle aspect in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`translation_x`**
    - Converts a list of x-axis translation values into float values, aiding in the horizontal movement control in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`translation_y`**
    - Converts a list of y-axis translation values into float values, facilitating vertical movement adjustments in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`translation_z`**
    - Transforms a list of z-axis translation values into float values, enabling depth adjustments in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`rotation_x`**
    - Converts a list of x-axis rotation values into float values, allowing for precise control over the rotation around the x-axis in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`rotation_y`**
    - Transforms a list of y-axis rotation values into float values, aiding in the control over rotation around the y-axis in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`rotation_z`**
    - Converts a list of z-axis rotation values into float values, enabling detailed adjustments to the rotation around the z-axis in animation keyframing.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`zoom`**
    - Comfy dtype: `FLOATS`
    - The converted float value for zoom, used in defining the zoom level in animation keyframing.
    - Python dtype: `float`
- **`angle`**
    - Comfy dtype: `FLOATS`
    - The converted float value for angle, used in setting the angle orientation in animation keyframing.
    - Python dtype: `float`
- **`translation_x`**
    - Comfy dtype: `FLOATS`
    - The converted float value for x-axis translation, used in horizontal movement control in animation keyframing.
    - Python dtype: `float`
- **`translation_y`**
    - Comfy dtype: `FLOATS`
    - The converted float value for y-axis translation, used in vertical movement control in animation keyframing.
    - Python dtype: `float`
- **`translation_z`**
    - Comfy dtype: `FLOATS`
    - The converted float value for z-axis translation, used in depth adjustment in animation keyframing.
    - Python dtype: `float`
- **`rotation_x`**
    - Comfy dtype: `FLOATS`
    - The converted float value for x-axis rotation, used in controlling rotation around the x-axis in animation keyframing.
    - Python dtype: `float`
- **`rotation_y`**
    - Comfy dtype: `FLOATS`
    - The converted float value for y-axis rotation, used in controlling rotation around the y-axis in animation keyframing.
    - Python dtype: `float`
- **`rotation_z`**
    - Comfy dtype: `FLOATS`
    - The converted float value for z-axis rotation, used in controlling rotation around the z-axis in animation keyframing.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OPAC2Floats:
    """
        Converts a LIST input to FLOATS type
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "zoom": ("LIST", {}),
                "angle": ("LIST", {}),
                "translation_x": ("LIST", {}),
                "translation_y": ("LIST", {}),
                "translation_z": ("LIST", {}),
                "rotation_x": ("LIST", {}),
                "rotation_y": ("LIST", {}),
                "rotation_z": ("LIST", {})
            }
        }
    
    RETURN_TYPES = ("FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS", "FLOATS")
    RETURN_NAMES = ("zoom", "angle", "translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z")
    FUNCTION = "convert"
    CATEGORY = "SALT/Animation/Keyframing"

    def convert(self, **kwargs):
        return (
                kwargs.get("zoom", [0]),
                kwargs.get("angle", [0]),
                kwargs.get("translation_x", [0]),
                kwargs.get("translation_y", [0]),
                kwargs.get("translation_z", [0]),
                kwargs.get("rotation_x", [0]),
                kwargs.get("rotation_y", [0]),
                kwargs.get("rotation_z", [0])
            )

```
