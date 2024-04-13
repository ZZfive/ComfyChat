# LLM Simple Web Page Reader
## Documentation
- Class name: `LLMSimpleWebPageReader`
- Category: `SALT/Llama-Index/Readers`
- Output node: `False`

The LLMSimpleWebPageReader node is designed for reading and extracting content from web pages. It supports multiple URLs and can optionally convert HTML content to text, facilitating the ingestion of web page data into language models or other processing pipelines.
## Input types
### Required
- **`url_1`**
    - The primary URL from which to read and extract content. This is a required parameter to initiate the reading process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`url_2`**
    - An optional secondary URL for reading and extracting additional content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_3`**
    - An optional third URL for reading and extracting additional content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_4`**
    - An optional fourth URL for reading and extracting additional content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`html_to_text`**
    - A boolean flag indicating whether to convert HTML content to plain text. This can be useful for text processing or analysis.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The extracted content from the provided URLs, potentially converted to text based on the html_to_text flag.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSimpleWebPageReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_web"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web(self, url_1, url_2=None, url_3=None, url_4=None, html_to_text=True):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")
        
        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())
        
        urls = [url for url in urls if valid_url(url)]

        print("Valided URLs:", urls)

        documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(urls)
        return (documents,)

```
