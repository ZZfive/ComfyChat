# 🪛 Pipe to/edit any
## Documentation
- Class name: `Pipe to_edit any [Crystools]`
- Category: `crystools 🪛/Pipe`
- Output node: `False`

The 'Pipe to/edit any' node is designed to facilitate the modification or updating of data flowing through a pipeline. It allows for the selective editing of up to six arbitrary data elements, enabling users to easily modify or replace these elements as needed within the flow of data.
## Input types
### Required
### Optional
- **`CPipeAny`**
    - Represents the original data elements to be potentially modified. It serves as a baseline for the modifications applied by the node, allowing for selective editing of the data elements.
    - Comfy dtype: `CPipeAny`
    - Python dtype: `Tuple[Any, Any, Any, Any, Any, Any]`
- **`any_1`**
    - An optional data element that can replace or modify the first element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_2`**
    - An optional data element that can replace or modify the second element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_3`**
    - An optional data element that can replace or modify the third element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_4`**
    - An optional data element that can replace or modify the fourth element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_5`**
    - An optional data element that can replace or modify the fifth element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_6`**
    - An optional data element that can replace or modify the sixth element of the original data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`cpipeany`**
    - Comfy dtype: `CPipeAny`
    - A list containing the modified or updated data elements, reflecting any changes made through the node.
    - Python dtype: `List[Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CPipeToAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                CLASSES.CPIPE_ANY_TYPE.value: (CLASSES.CPIPE_ANY_TYPE.value,),
                "any_1": (any,),
                "any_2": (any,),
                "any_3": (any,),
                "any_4": (any,),
                "any_5": (any,),
                "any_6": (any,),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.PIPE.value
    RETURN_TYPES = (CLASSES.CPIPE_ANY_TYPE.value,)

    FUNCTION = "execute"

    def execute(self, CPipeAny=None, any_1=None, any_2=None, any_3=None, any_4=None, any_5=None, any_6=None):
        any_1_original = None
        any_2_original = None
        any_3_original = None
        any_4_original = None
        any_5_original = None
        any_6_original = None

        if CPipeAny != None:
            any_1_original, any_2_original, any_3_original, any_4_original, any_5_original, any_6_original = CPipeAny

        CAnyPipeMod = []

        CAnyPipeMod.append(any_1 if any_1 is not None else any_1_original)
        CAnyPipeMod.append(any_2 if any_2 is not None else any_2_original)
        CAnyPipeMod.append(any_3 if any_3 is not None else any_3_original)
        CAnyPipeMod.append(any_4 if any_4 is not None else any_4_original)
        CAnyPipeMod.append(any_5 if any_5 is not None else any_5_original)
        CAnyPipeMod.append(any_6 if any_6 is not None else any_6_original)

        return (CAnyPipeMod,)

```
