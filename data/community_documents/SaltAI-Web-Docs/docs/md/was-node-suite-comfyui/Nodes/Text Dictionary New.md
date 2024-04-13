# Text Dictionary New
## Documentation
- Class name: `Text Dictionary New`
- Category: `WAS Suite/Text`
- Output node: `False`

This node is designed to create a new dictionary from a set of key-value pairs provided as inputs. It allows for the dynamic construction of dictionaries within a workflow, enabling the storage and manipulation of data in a structured format.
## Input types
### Required
- **`key_1`**
    - The first key to be included in the new dictionary. This input, along with its corresponding value, forms the first key-value pair in the constructed dictionary.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`value_1`**
    - The value associated with the first key in the new dictionary. This input is crucial for establishing the initial content of the dictionary.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`key_2`**
    - An optional second key for the dictionary. If provided, it adds another key-value pair to the dictionary, expanding its content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`value_2`**
    - The value associated with the optional second key, contributing to the dictionary's expanded content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`key_3`**
    - An optional third key for the dictionary, further extending its structure with additional data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`value_3`**
    - The value for the optional third key, enhancing the dictionary with more detailed information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`key_4`**
    - An optional fourth key, allowing for even more data to be structured within the dictionary.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`value_4`**
    - The corresponding value for the fourth key, adding further detail to the dictionary's data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`key_5`**
    - The fifth optional key, providing the possibility to include more comprehensive data in the dictionary.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`value_5`**
    - The value associated with the fifth key, completing the dictionary's structure with a wide range of information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`dict`**
    - Comfy dtype: `DICT`
    - The newly constructed dictionary, formed from the provided key-value pairs, enabling structured data storage and manipulation.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WAS_Dictionary_New:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_1": ("STRING", {"default":"", "multiline": False}),
                "value_1": ("STRING", {"default":"", "multiline": False}),
            },
            "optional": {
                "key_2": ("STRING", {"default":"", "multiline": False}),
                "value_2": ("STRING", {"default":"", "multiline": False}),
                "key_3": ("STRING", {"default":"", "multiline": False}),
                "value_3": ("STRING", {"default":"", "multiline": False}),
                "key_4": ("STRING", {"default":"", "multiline": False}),
                "value_4": ("STRING", {"default":"", "multiline": False}),
                "key_5": ("STRING", {"default":"", "multiline": False}),
                "value_5": ("STRING", {"default":"", "multiline": False}),
            }
        }
    RETURN_TYPES = ("DICT",)
    FUNCTION = "dictionary_new"

    CATEGORY = "WAS Suite/Text"

    def append_to_dictionary(self, dictionary, key, value):
        if key is not None and key != "":
            dictionary[key] = value
        return dictionary

    def dictionary_new(self, key_1, value_1, key_2, value_2, key_3, value_3, key_4, value_4, key_5, value_5):
        dictionary = {}
        dictionary = self.append_to_dictionary(dictionary, key_1, value_1)
        dictionary = self.append_to_dictionary(dictionary, key_2, value_2)
        dictionary = self.append_to_dictionary(dictionary, key_3, value_3)
        dictionary = self.append_to_dictionary(dictionary, key_4, value_4)
        dictionary = self.append_to_dictionary(dictionary, key_5, value_5)
        return (dictionary, )

```
