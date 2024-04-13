# 🪛 List of any
## Documentation
- Class name: `List of any [Crystools]`
- Category: `crystools 🪛/List`
- Output node: `False`

This node is designed to aggregate multiple inputs of any type into a single list, facilitating the handling and manipulation of diverse data types together. It serves as a versatile tool for collecting a variable number of inputs into a unified structure.
## Input types
### Required
### Optional
- **`any_1`**
    - Represents the first optional input of any type that can be included in the list. Its inclusion enhances the node's flexibility in aggregating diverse data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_2`**
    - Serves as the second optional input, allowing for the aggregation of additional diverse data into the list.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_3`**
    - Acts as the third optional input, further extending the node's capability to compile a varied set of data into a single list.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_4`**
    - Functions as the fourth optional input, contributing to the node's versatility in collecting different types of data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_5`**
    - Represents the fifth optional input, enabling the inclusion of more varied data into the aggregated list.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_6`**
    - Serves as the sixth optional input, further broadening the scope of data types that can be compiled into the list.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_7`**
    - Acts as the seventh optional input, enhancing the node's ability to aggregate a wide range of data types.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`any_8`**
    - Functions as the eighth optional input, maximizing the node's capacity to collect diverse data into a unified list.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`any_list`**
    - Comfy dtype: `COMBO[STRING]`
    - The output is a list containing all the provided inputs, regardless of their type, facilitating further operations on a unified data structure.
    - Python dtype: `List[Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CListAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "any_1": (any,),
                "any_2": (any,),
                "any_3": (any,),
                "any_4": (any,),
                "any_5": (any,),
                "any_6": (any,),
                "any_7": (any,),
                "any_8": (any,),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LIST.value
    RETURN_TYPES = (any,),
    RETURN_NAMES = ("any_list",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "execute"

    def execute(self,
                any_1=None,
                any_2=None,
                any_3=None,
                any_4=None,
                any_5=None,
                any_6=None,
                any_7=None,
                any_8=None):

        list_any = []

        if any_1 is not None:
            try:
                list_any.append(any_1)
            except Exception as e:
                logger.warn(e)
        if any_2 is not None:
            try:
                list_any.append(any_2)
            except Exception as e:
                logger.warn(e)
        if any_3 is not None:
            try:
                list_any.append(any_3)
            except Exception as e:
                logger.warn(e)
        if any_4 is not None:
            try:
                list_any.append(any_4)
            except Exception as e:
                logger.warn(e)
        if any_5 is not None:
            try:
                list_any.append(any_5)
            except Exception as e:
                logger.warn(e)
        if any_6 is not None:
            try:
                list_any.append(any_6)
            except Exception as e:
                logger.warn(e)
        if any_7 is not None:
            try:
                list_any.append(any_7)
            except Exception as e:
                logger.warn(e)
        if any_8 is not None:
            try:
                list_any.append(any_8)
            except Exception as e:
                logger.warn(e)

        # yes, double brackets are needed because of the OUTPUT_IS_LIST... ¯\_(ツ)_/¯
        return [[list_any]]

```
