# Math Expression (mtb)
## Documentation
- Class name: `Math Expression (mtb)`
- Category: `mtb/math`
- Output node: `False`

This node is designed to evaluate simple math expressions provided as strings. It supports basic arithmetic operations and can dynamically replace placeholders within the expression with specified values.
## Input types
### Required
- **`expression`**
    - The math expression to be evaluated. Supports basic arithmetic operations and placeholders for dynamic value substitution.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`result (float)`**
    - Comfy dtype: `FLOAT`
    - The floating-point result of the evaluated math expression.
    - Python dtype: `float`
- **`result (int)`**
    - Comfy dtype: `INT`
    - The integer result of the evaluated math expression, obtained by converting the float result to an integer.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_MathExpression:
    """Node to evaluate a simple math expression string"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"default": "", "multiline": True}),
            }
        }

    FUNCTION = "eval_expression"
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result (float)", "result (int)")
    CATEGORY = "mtb/math"
    DESCRIPTION = "evaluate a simple math expression string (!! Fallsback to eval)"

    def eval_expression(self, expression, **kwargs):
        import math
        from ast import literal_eval

        for key, value in kwargs.items():
            print(f"Replacing placeholder <{key}> with value {value}")
            expression = expression.replace(f"<{key}>", str(value))

        result = -1
        try:
            result = literal_eval(expression)
        except SyntaxError as e:
            raise ValueError(
                f"The expression syntax is wrong '{expression}': {e}"
            ) from e

        except ValueError:
            try:
                expression = expression.replace("^", "**")
                result = eval(expression)
            except Exception as e:
                # Handle any other exceptions and provide a meaningful error message
                raise ValueError(
                    f"Error evaluating expression '{expression}': {e}"
                ) from e

        return (result, int(result))

```
