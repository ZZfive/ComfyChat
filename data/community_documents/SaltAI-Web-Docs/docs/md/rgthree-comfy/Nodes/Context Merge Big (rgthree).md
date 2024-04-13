# Context Merge Big (rgthree)
## Documentation
- Class name: `Context Merge Big (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Context Merge Big node is designed to merge multiple context objects into a single, comprehensive context. It prioritizes later contexts over earlier ones, allowing for dynamic updates and overrides of context information. This functionality is crucial for managing and updating the state of context-driven processes in a flexible and efficient manner.
## Input types
### Required
### Optional
- **`ctx_01`**
    - Represents the first context object to be merged. It serves as the base context, which can be overridden by subsequent contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_02`**
    - Represents the second context object to be merged. It can override values in the first context.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_03`**
    - Represents the third context object to be merged. It can override values in the first and second contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_04`**
    - Represents the fourth context object to be merged. It can override values in the first, second, and third contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_05`**
    - Represents the fifth context object to be merged. It has the highest priority and can override values in all previous contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
## Output types
- **`CONTEXT`**
    - Comfy dtype: `RGTHREE_CONTEXT`
    - The merged context's comprehensive configuration, encompassing all aspects of the context.
    - Python dtype: `dict`
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The merged context's model configuration.
    - Python dtype: `str`
- **`CLIP`**
    - Comfy dtype: `CLIP`
    - The merged context's CLIP model configuration.
    - Python dtype: `str`
- **`VAE`**
    - Comfy dtype: `VAE`
    - The merged context's VAE model configuration.
    - Python dtype: `str`
- **`POSITIVE`**
    - Comfy dtype: `CONDITIONING`
    - The merged context's positive conditioning.
    - Python dtype: `str`
- **`NEGATIVE`**
    - Comfy dtype: `CONDITIONING`
    - The merged context's negative conditioning.
    - Python dtype: `str`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The merged context's latent space configuration.
    - Python dtype: `str`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The merged context's image data.
    - Python dtype: `str`
- **`SEED`**
    - Comfy dtype: `INT`
    - The merged context's seed for random number generation.
    - Python dtype: `int`
- **`STEPS`**
    - Comfy dtype: `INT`
    - The merged context's number of steps for the process.
    - Python dtype: `int`
- **`STEP_REFINER`**
    - Comfy dtype: `INT`
    - The merged context's step refiner configuration.
    - Python dtype: `int`
- **`CFG`**
    - Comfy dtype: `FLOAT`
    - The merged context's CFG scale.
    - Python dtype: `float`
- **`CKPT_NAME`**
    - Comfy dtype: `COMBO[STRING]`
    - The merged context's checkpoint name for model loading.
    - Python dtype: `str`
- **`SAMPLER`**
    - Comfy dtype: `COMBO[STRING]`
    - The merged context's sampler configuration.
    - Python dtype: `str`
- **`SCHEDULER`**
    - Comfy dtype: `COMBO[STRING]`
    - The merged context's scheduler configuration.
    - Python dtype: `str`
- **`CLIP_WIDTH`**
    - Comfy dtype: `INT`
    - The merged context's width for CLIP model processing.
    - Python dtype: `int`
- **`CLIP_HEIGHT`**
    - Comfy dtype: `INT`
    - The merged context's height for CLIP model processing.
    - Python dtype: `int`
- **`TEXT_POS_G`**
    - Comfy dtype: `STRING`
    - The merged context's global positive text.
    - Python dtype: `str`
- **`TEXT_POS_L`**
    - Comfy dtype: `STRING`
    - The merged context's local positive text.
    - Python dtype: `str`
- **`TEXT_NEG_G`**
    - Comfy dtype: `STRING`
    - The merged context's global negative text.
    - Python dtype: `str`
- **`TEXT_NEG_L`**
    - Comfy dtype: `STRING`
    - The merged context's local negative text.
    - Python dtype: `str`
- **`MASK`**
    - Comfy dtype: `MASK`
    - The merged context's mask configuration.
    - Python dtype: `str`
- **`CONTROL_NET`**
    - Comfy dtype: `CONTROL_NET`
    - The merged context's control net configuration.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RgthreeContextMergeBig:
  """The Context Merge Big node."""

  NAME = get_name("Context Merge Big")
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": {
        "ctx_01": ("RGTHREE_CONTEXT",),
        "ctx_02": ("RGTHREE_CONTEXT",),
        "ctx_03": ("RGTHREE_CONTEXT",),
        "ctx_04": ("RGTHREE_CONTEXT",),
        "ctx_05": ("RGTHREE_CONTEXT",),
      },
    }

  RETURN_TYPES = ALL_CTX_RETURN_TYPES
  RETURN_NAMES = ALL_CTX_RETURN_NAMES
  FUNCTION = "merge"

  def merge(self, ctx_01=None, ctx_02=None, ctx_03=None, ctx_04=None, ctx_05=None):
    """Merges any non-null passed contexts; later ones overriding earlier.
    """
    ctx = merge_new_context(ctx_01, ctx_02, ctx_03, ctx_04, ctx_05)

    return get_context_return_tuple(ctx)

```
