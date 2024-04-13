# Any Switch (rgthree)
## Documentation
- Class name: `Any Switch (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Any Switch node is designed to select and output the first non-empty item from a set of inputs. It abstracts the process of checking multiple inputs for validity and provides a streamlined way to retrieve the first valid value, enhancing flexibility and efficiency in data processing.
## Input types
### Required
### Optional
- **`any_01`**
    - Represents the first optional input that can be of any type. The node checks this input first for non-emptiness and outputs it if valid.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_02`**
    - Represents the second optional input of any type. It is checked for non-emptiness and outputted if the previous input is empty and this is valid.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_03`**
    - The third optional input of any type. It is considered for output if all previous inputs are empty and this input is valid.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_04`**
    - This is the fourth optional input, which can be of any type. It is selected for output if all preceding inputs are empty and this is valid.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_05`**
    - The fifth and last optional input of any type. It is chosen for output if all other inputs are empty and this input is valid.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`*`**
    - Comfy dtype: `*`
    - Outputs the first non-empty item from the provided inputs. This allows for dynamic selection based on input validity.
    - Python dtype: `Any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RgthreeAnySwitch:
  """The any switch. """

  NAME = get_name("Any Switch")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": {
        "any_01": (any_type,),
        "any_02": (any_type,),
        "any_03": (any_type,),
        "any_04": (any_type,),
        "any_05": (any_type,),
      },
    }

  RETURN_TYPES = (any_type,)
  RETURN_NAMES = ('*',)
  FUNCTION = "switch"

  def switch(self, any_01=None, any_02=None, any_03=None, any_04=None, any_05=None):
    """Chooses the first non-empty item to output."""
    any_value = None
    if not is_none(any_01):
      any_value = any_01
    elif not is_none(any_02):
      any_value = any_02
    elif not is_none(any_03):
      any_value = any_03
    elif not is_none(any_04):
      any_value = any_04
    elif not is_none(any_05):
      any_value = any_05
    return (any_value,)

```
