# Context Switch Big (rgthree)
## Documentation
- Class name: `Context Switch Big (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Context Switch Big node is designed to manage multiple context inputs, selecting the first non-empty context among them for output. This functionality allows for dynamic context switching based on the availability of data, facilitating flexible and efficient context management in complex workflows.
## Input types
### Required
### Optional
- **`ctx_01`**
    - Represents an optional context input. If non-empty, it can be selected as the output context, depending on its order and the emptiness of preceding contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_02`**
    - Represents an optional context input. If non-empty, it can be selected as the output context, depending on its order and the emptiness of preceding contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_03`**
    - Represents an optional context input. If non-empty, it can be selected as the output context, depending on its order and the emptiness of preceding contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_04`**
    - Represents an optional context input. If non-empty, it can be selected as the output context, depending on its order and the emptiness of preceding contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
- **`ctx_05`**
    - Represents an optional context input. If non-empty, it can be selected as the output context, depending on its order and the emptiness of preceding contexts.
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Python dtype: `dict`
## Output types
- **`CONTEXT`**
    - Comfy dtype: `RGTHREE_CONTEXT`
    - Outputs the selected non-empty context among the inputs, facilitating dynamic context management.
    - Python dtype: `dict`
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - Outputs the model information from the selected context, if available.
    - Python dtype: `dict`
- **`CLIP`**
    - Comfy dtype: `CLIP`
    - Outputs the clip information from the selected context, if available.
    - Python dtype: `dict`
- **`VAE`**
    - Comfy dtype: `VAE`
    - Outputs the VAE information from the selected context, if available.
    - Python dtype: `dict`
- **`POSITIVE`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive conditioning from the selected context, if available.
    - Python dtype: `dict`
- **`NEGATIVE`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative conditioning from the selected context, if available.
    - Python dtype: `dict`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - Outputs the latent information from the selected context, if available.
    - Python dtype: `dict`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - Outputs the images from the selected context, if available.
    - Python dtype: `dict`
- **`SEED`**
    - Comfy dtype: `INT`
    - Outputs the seed value from the selected context, if available.
    - Python dtype: `dict`
- **`STEPS`**
    - Comfy dtype: `INT`
    - Outputs the steps value from the selected context, if available.
    - Python dtype: `dict`
- **`STEP_REFINER`**
    - Comfy dtype: `INT`
    - Outputs the step refiner value from the selected context, if available.
    - Python dtype: `dict`
- **`CFG`**
    - Comfy dtype: `FLOAT`
    - Outputs the CFG value from the selected context, if available.
    - Python dtype: `dict`
- **`CKPT_NAME`**
    - Comfy dtype: `COMBO[STRING]`
    - Outputs the checkpoint name from the selected context, if available.
    - Python dtype: `dict`
- **`SAMPLER`**
    - Comfy dtype: `COMBO[STRING]`
    - Outputs the sampler information from the selected context, if available.
    - Python dtype: `dict`
- **`SCHEDULER`**
    - Comfy dtype: `COMBO[STRING]`
    - Outputs the scheduler information from the selected context, if available.
    - Python dtype: `dict`
- **`CLIP_WIDTH`**
    - Comfy dtype: `INT`
    - Outputs the clip width from the selected context, if available.
    - Python dtype: `dict`
- **`CLIP_HEIGHT`**
    - Comfy dtype: `INT`
    - Outputs the clip height from the selected context, if available.
    - Python dtype: `dict`
- **`TEXT_POS_G`**
    - Comfy dtype: `STRING`
    - Outputs the global positive text from the selected context, if available.
    - Python dtype: `dict`
- **`TEXT_POS_L`**
    - Comfy dtype: `STRING`
    - Outputs the local positive text from the selected context, if available.
    - Python dtype: `dict`
- **`TEXT_NEG_G`**
    - Comfy dtype: `STRING`
    - Outputs the global negative text from the selected context, if available.
    - Python dtype: `dict`
- **`TEXT_NEG_L`**
    - Comfy dtype: `STRING`
    - Outputs the local negative text from the selected context, if available.
    - Python dtype: `dict`
- **`MASK`**
    - Comfy dtype: `MASK`
    - Outputs the mask information from the selected context, if available.
    - Python dtype: `dict`
- **`CONTROL_NET`**
    - Comfy dtype: `CONTROL_NET`
    - Outputs the control net information from the selected context, if available.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RgthreeContextSwitchBig:
  """The Context Switch Big node."""

  NAME = get_name("Context Switch Big")
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
  FUNCTION = "switch"

  def switch(self, ctx_01=None, ctx_02=None, ctx_03=None, ctx_04=None, ctx_05=None):
    """Chooses the first non-empty Context to output.
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
    return get_context_return_tuple(ctx)

```
