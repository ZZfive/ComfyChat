# Float to Integer
## Documentation
- Class name: `JWFloatToInteger`
- Category: `jamesWalker55`
- Output node: `False`

This node converts a floating-point number to an integer, offering different rounding modes to accommodate various numerical precision requirements.
## Input types
### Required
- **`value`**
    - The floating-point number to be converted to an integer. This parameter is the primary input for the conversion process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mode`**
    - Specifies the rounding mode to be used for the conversion: 'round' for standard rounding, 'floor' for rounding down, and 'ceiling' for rounding up. This allows for flexible control over the conversion outcome.
    - Comfy dtype: `['round', 'floor', 'ceiling']`
    - Python dtype: `Literal['round', 'floor', 'ceiling']`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The resulting integer after converting the floating-point number according to the specified rounding mode.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
        @register_node(identifier, display_name)
        class _:
            CATEGORY = category
            INPUT_TYPES = lambda: {"required": required_inputs}
            RETURN_TYPES = tuple(return_types)
            OUTPUT_NODE = output_node
            FUNCTION = "execute"

            def execute(self, *args, **kwargs):
                return func(*args, **kwargs)

```
