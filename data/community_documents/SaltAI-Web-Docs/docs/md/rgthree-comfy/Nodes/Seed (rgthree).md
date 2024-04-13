# Seed (rgthree)
## Documentation
- Class name: `Seed (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Seed node is designed to simply return the seed value that is passed to it. This functionality is crucial for ensuring consistency and reproducibility in operations that rely on random number generation, by allowing users to specify a seed value.
## Input types
### Required
- **`seed`**
    - The 'seed' parameter allows users to specify a seed value for random number generation, ensuring consistent and reproducible results across runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`SEED`**
    - Comfy dtype: `INT`
    - Returns the seed value that was passed to the node.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - PhotoMaker_Generation
    - KSampler Adv. (Efficient)
    - KSampler SDXL (Eff.)
    - OneButtonPrompt



## Source code
```python
class RgthreeSeed:
  """See node."""

  NAME = get_name('Seed')
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "seed": ("INT", {
          "default": 0,
          "min": -1125899906842624,
          "max": 1125899906842624
        }),
      },
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("SEED",)
  FUNCTION = "main"

  def main(self, seed=0):
    """Returns the passed seed on execution."""
    return (seed,)

```
