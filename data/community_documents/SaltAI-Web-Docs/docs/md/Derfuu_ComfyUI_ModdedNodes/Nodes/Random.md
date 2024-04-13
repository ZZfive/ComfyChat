# Random
## Documentation
- Class name: `Random`
- Category: `Derfuu_Nodes/Functions`
- Output node: `False`

The Random node generates a random floating-point number within a specified range. It abstracts the complexity of random number generation, providing a simple interface for obtaining a random value between two bounds.
## Input types
### Required
- **`Value_A`**
    - Specifies the lower bound of the random number range. It sets the minimum possible value that can be generated.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`Value_B`**
    - Defines the upper bound of the random number range. It sets the maximum possible value that can be generated.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The random floating-point number generated within the specified range.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RandomValue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": field.FLOAT,
                "Value_B": field.FLOAT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_rand"

    CATEGORY = TREE_FUNCTIONS

    def get_rand(self, Value_A, Value_B):
        value = numpy.random.uniform(Value_A, Value_B)
        return (value,)

```
