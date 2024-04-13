# LLM RSS Reader
## Documentation
- Class name: `LLMRssReaderNode`
- Category: `SALT/Llama-Index/Readers`
- Output node: `False`

The LLMRssReaderNode is designed to fetch and process RSS feed data from provided URLs, transforming the raw feed into a structured format suitable for further analysis or processing.
## Input types
### Required
- **`url_1`**
    - The primary URL from which the RSS feed will be read. This is a mandatory input to ensure the node has at least one source to process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`url_2`**
    - An optional secondary URL for reading an RSS feed, allowing for the aggregation of data from multiple sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_3`**
    - An optional tertiary URL for reading an RSS feed, further expanding the node's capability to aggregate data from various sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`url_4`**
    - An optional quaternary URL for reading an RSS feed, maximizing the node's potential for data aggregation from multiple sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The processed RSS feed data, structured and ready for further analysis or integration into downstream processes.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMRssReaderNode:
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

    FUNCTION = "read_rss"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_rss(self, url_1, url_2=None, url_3=None, url_4=None):
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

        documents = RssReader().load_data(urls)
        return (documents,)

```
