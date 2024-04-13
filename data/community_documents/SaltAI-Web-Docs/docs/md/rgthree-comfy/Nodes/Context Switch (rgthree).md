# Context Switch (rgthree)
## Documentation
- Class name: `Context Switch (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Context Switch node is designed to select and output the first non-empty context from a series of provided contexts. It ensures seamless integration and compatibility with other context nodes, facilitating the dynamic selection of contexts based on their content.
## Input types
### Required
### Optional
- **`ctx_01`**
    - Represents one of the potential contexts to be evaluated and selected if it's the first non-empty context.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_02`**
    - Represents another potential context to be evaluated and selected if it's the first non-empty context following an empty 'ctx_01'.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_03`**
    - Serves as a potential context to be selected, evaluated after 'ctx_01' and 'ctx_02' if they are empty.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_04`**
    - Another context option, evaluated for selection if preceding contexts ('ctx_01', 'ctx_02', 'ctx_03') are empty.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_05`**
    - The last in the series of potential contexts to be evaluated and possibly selected if all preceding contexts are empty.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
## Output types
- **`CONTEXT`**
    - Comfy dtype: `RGTHREE_CONTEXT`
    - The original context keys selected from the provided options.
    - Python dtype: `tuple`
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - Model information from the selected context.
    - Python dtype: `tuple`
- **`CLIP`**
    - Comfy dtype: `CLIP`
    - Clip information from the selected context.
    - Python dtype: `tuple`
- **`VAE`**
    - Comfy dtype: `VAE`
    - VAE information from the selected context.
    - Python dtype: `tuple`
- **`POSITIVE`**
    - Comfy dtype: `CONDITIONING`
    - Positive conditioning from the selected context.
    - Python dtype: `tuple`
- **`NEGATIVE`**
    - Comfy dtype: `CONDITIONING`
    - Negative conditioning from the selected context.
    - Python dtype: `tuple`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - Latent information from the selected context.
    - Python dtype: `tuple`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - Image information from the selected context.
    - Python dtype: `tuple`
- **`SEED`**
    - Comfy dtype: `INT`
    - Seed information from the selected context.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RgthreeContextSwitch:
  """The initial Context Switch node.

  For now, this will remain as-is but is otherwise backwards compatible with other Context nodes
  outputs.
  """

  NAME = get_name("Context Switch")
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
  FUNCTION = "switch"

  def switch(self, ctx_01=None, ctx_02=None, ctx_03=None, ctx_04=None, ctx_05=None):
    """Chooses the first non-empty Context to output.

    As of right now, this returns the "original" context. We could expand it, or create another
    "Context Big Switch" and have all the outputs...
    """
    ctx = None
    if not is_context_empty(ctx_01):
      ctx = ctx_01
    elif not is_context_empty(ctx_02):
      ctx = ctx_02
    elif not is_context_empty(ctx_03):
      ctx = ctx_03
    elif not is_context_empty(ctx_04):
      ctx = ctx_04
    elif not is_context_empty(ctx_05):
      ctx = ctx_05
    return get_orig_context_return_tuple(ctx)

```
