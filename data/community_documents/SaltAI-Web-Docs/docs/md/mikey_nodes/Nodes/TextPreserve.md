# Text Preserve (Mikey)
## Documentation
- Class name: `TextPreserve`
- Category: `Mikey/Text`
- Output node: `True`

The TextPreserve node is designed to maintain the original text input while applying specific transformations based on search and replace patterns, wildcard syntax, and workflow modifications. It aims to preserve the initial text input for further use or reference, while also updating the workflow's nodes with new widget values reflecting the original and transformed texts.
## Input types
### Required
- **`text`**
    - The original text input that will be preserved and potentially transformed based on the provided patterns and syntax.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`result_text`**
    - The text resulting from applying the search and replace operations and wildcard syntax to the original text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The transformed text after applying search and replace operations and wildcard syntax.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TextPreserve:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'text': ('STRING', {'multiline': True, 'default': 'Input Text Here', 'dynamicPrompts': False}),
                             'result_text': ('STRING', {'multiline': True, 'default': 'Result Text Here (will be replaced)'})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'
    OUTPUT_NODE = True

    CATEGORY = 'Mikey/Text'

    def process(self, text, result_text, unique_id=None, extra_pnginfo=None, prompt=None):
        preserve_text = text
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(text):
            text = text.replace(m.group(0), repl(m))
        prompt.get(str(unique_id))['inputs']['text'] = preserve_text
        for i, node_dict in enumerate(extra_pnginfo['workflow']['nodes']):
            if node_dict['id'] == int(unique_id):
                node_dict['widgets_values'] = [preserve_text, text]
                extra_pnginfo['workflow']['nodes'][i] = node_dict
        prompt.get(str(unique_id))['inputs']['result_text'] = text
        return (text,)

```
