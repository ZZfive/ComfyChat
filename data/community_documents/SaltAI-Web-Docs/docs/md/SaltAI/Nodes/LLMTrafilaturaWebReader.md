# LLM Trafilatura Web Reader
## Documentation
- Class name: `LLMTrafilaturaWebReader`
- Category: `SALT/Llama-Index/Readers`
- Output node: `False`

The LLMTrafilaturaWebReader node is designed for reading and extracting content from web pages using the Trafilatura library. It accepts multiple URLs as input and returns the extracted documents, facilitating the ingestion of web content into the LLM ecosystem.
## Input types
### Required
- **`url_1`**
    - The primary URL from which to extract content. This is a required parameter to initiate the content extraction process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`url_2`**
    - An optional secondary URL for content extraction, allowing for the processing of multiple web pages in a single operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_3`**
    - An optional tertiary URL for content extraction, further expanding the node's capability to process multiple web pages simultaneously.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_4`**
    - An optional quaternary URL for content extraction, enabling the node to handle up to four web pages in one execution.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The extracted documents from the provided URLs, ready for further processing or analysis within the LLM ecosystem.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMTrafilaturaWebReader:
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
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web_trafilatura(self, url_1, url_2=None, url_3=None, url_4=None):
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

        documents = TrafilaturaWebReader().load_data(urls)
        return (documents,)

```
