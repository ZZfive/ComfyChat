# 🪛 List of strings
## Documentation
- Class name: `List of strings [Crystools]`
- Category: `crystools 🪛/List`
- Output node: `False`

This node is designed to aggregate multiple string inputs into a single list or a concatenated string, based on the provided delimiter. It facilitates the handling and manipulation of string collections within a workflow, allowing for flexible data structuring and processing.
## Input types
### Required
### Optional
- **`string_1`**
    - The first string to be considered for aggregation. It contributes to the formation of the list or the concatenated string output, depending on the delimiter. Its presence or absence directly influences whether the output list will contain this string, affecting the overall data structure.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_2`**
    - The second string to be considered for aggregation. It adds to the list or concatenated string, enhancing the data structure. Its inclusion adds another element to the output list or extends the concatenated string, impacting the final data representation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_3`**
    - The third string input for aggregation. It plays a role in enriching the list or concatenated string output. Its contribution is crucial for expanding the output list or concatenated string, thereby enriching the data available for further processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_4`**
    - The fourth string input for aggregation. It further contributes to the complexity of the list or concatenated string. Including this string diversifies the output, allowing for more nuanced data manipulation and analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_5`**
    - The fifth string input for aggregation. It adds depth to the list or concatenated string, allowing for more detailed data manipulation. Its addition provides a broader range of data within the output, facilitating more comprehensive analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_6`**
    - The sixth string input for aggregation. It enhances the list or concatenated string with additional data. This input further enriches the output, broadening the scope of data available for subsequent operations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_7`**
    - The seventh string input for aggregation. It contributes to the diversity of the list or concatenated string. Its presence introduces more variety into the output, enabling more versatile data handling.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_8`**
    - The eighth string input for aggregation. It completes the set of data for the list or concatenated string. This final input ensures the output is comprehensive, covering a wider array of data points for processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - A string delimiter used to concatenate the input strings. If empty, the strings are aggregated into a list. This parameter dictates the format of the output, either as a single concatenated string or a list of strings, thus directly influencing the node's execution and the structure of the results.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`concatenated`**
    - Comfy dtype: `STRING`
    - The concatenated string formed by joining the input strings with the specified delimiter.
    - Python dtype: `str`
- **`list_string`**
    - Comfy dtype: `ListString`
    - A list containing the input strings that were not empty.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CListString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "string_1": STRING,
                "string_2": STRING,
                "string_3": STRING,
                "string_4": STRING,
                "string_5": STRING,
                "string_6": STRING,
                "string_7": STRING,
                "string_8": STRING,
                "delimiter": ("STRING", {"default": " "}),
            }
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LIST.value
    RETURN_TYPES = ("STRING", CLASSES.CLIST_STRING_TYPE.value,)
    RETURN_NAMES = (TEXTS.CONCAT.value, KEYS.LIST.value)
    OUTPUT_IS_LIST = (False, True, )

    FUNCTION = "execute"

    def execute(self,
                string_1=None,
                string_2=None,
                string_3=None,
                string_4=None,
                string_5=None,
                string_6=None,
                string_7=None,
                string_8=None,
                delimiter=""):

        list_str = []

        if string_1 is not None and string_1 != "":
            list_str.append(string_1)
        if string_2 is not None and string_2 != "":
            list_str.append(string_2)
        if string_3 is not None and string_3 != "":
            list_str.append(string_3)
        if string_4 is not None and string_4 != "":
            list_str.append(string_4)
        if string_5 is not None and string_5 != "":
            list_str.append(string_5)
        if string_6 is not None and string_6 != "":
            list_str.append(string_6)
        if string_7 is not None and string_7 != "":
            list_str.append(string_7)
        if string_8 is not None and string_8 != "":
            list_str.append(string_8)

        return delimiter.join(list_str), [list_str]

```
