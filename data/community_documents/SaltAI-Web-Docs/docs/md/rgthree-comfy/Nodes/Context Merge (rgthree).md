# Context Merge (rgthree)
## Documentation
- Class name: `Context Merge (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Context Merge node is designed to integrate multiple context objects into a single, unified context. By accepting up to five optional context inputs, it allows for the dynamic combination and overriding of context properties, facilitating flexible and efficient context management across different operations.
## Input types
### Required
### Optional
- **`ctx_01`**
    - Represents the first context object to be merged. If provided, its properties may be overridden by subsequent context objects.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_02`**
    - Represents the second context object to be merged, potentially overriding properties from the first context.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_03`**
    - The third context object in the merging sequence, which can override properties from the first and second contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_04`**
    - This is the fourth context object considered for merging, capable of overriding properties from earlier contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_05`**
    - The final context object in the merge sequence, having the highest priority in overriding properties from all previous contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
## Output types
- **`CONTEXT`**
    - Comfy dtype: `RGTHREE_CONTEXT`
    - The unified context resulting from the merge, incorporating elements from all provided contexts.
    - Python dtype: `dict`
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The merged model settings from the provided contexts.
    - Python dtype: `dict`
- **`CLIP`**
    - Comfy dtype: `CLIP`
    - The merged CLIP settings from the provided contexts.
    - Python dtype: `dict`
- **`VAE`**
    - Comfy dtype: `VAE`
    - The merged VAE settings from the provided contexts.
    - Python dtype: `dict`
- **`POSITIVE`**
    - Comfy dtype: `CONDITIONING`
    - The merged positive conditioning from the provided contexts.
    - Python dtype: `dict`
- **`NEGATIVE`**
    - Comfy dtype: `CONDITIONING`
    - The merged negative conditioning from the provided contexts.
    - Python dtype: `dict`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The merged latent settings from the provided contexts.
    - Python dtype: `dict`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The merged image settings from the provided contexts.
    - Python dtype: `dict`
- **`SEED`**
    - Comfy dtype: `INT`
    - The merged seed settings from the provided contexts.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RgthreeContextMerge:
  """The Context Merge Big node."""

  NAME = get_name("Context Merge")
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

  RETURN_TYPES = ORIG_CTX_RETURN_TYPES
  RETURN_NAMES = ORIG_CTX_RETURN_NAMES
  FUNCTION = "merge"

  def merge(self, ctx_01=None, ctx_02=None, ctx_03=None, ctx_04=None, ctx_05=None):
    """Merges any non-null passed contexts; later ones overriding earlier.
    """
    ctx = merge_new_context(ctx_01, ctx_02, ctx_03, ctx_04, ctx_05)

    return get_orig_context_return_tuple(ctx)

```
