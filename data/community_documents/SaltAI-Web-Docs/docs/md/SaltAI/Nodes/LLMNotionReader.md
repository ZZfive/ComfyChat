# LLM Notion Reader
## Documentation
- Class name: `LLMNotionReader`
- Category: `SALT/Llama-Index/Readers`
- Output node: `False`

The LLMNotionReader node is designed to fetch documents from Notion pages or databases using a Notion integration token. It abstracts the complexities of interacting with Notion's API, providing a straightforward way to retrieve content based on page IDs or a database ID.
## Input types
### Required
- **`notion_integration_token`**
    - This token is essential for authenticating and establishing a connection with Notion's API, enabling the retrieval of documents.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`page_ids`**
    - A comma-separated list of page IDs from which documents are to be retrieved. This allows for fetching content from specific Notion pages.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`database_id`**
    - An optional parameter specifying the ID of a Notion database from which documents are to be fetched. If provided, the node will retrieve documents from this database.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The output is a collection of documents fetched from the specified Notion pages or database.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMNotionReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "notion_integration_token": ("STRING", {}),
                "page_ids": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Page ID 1, Page ID 2"}),
                "database_id": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Database ID", "optional": True}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_notion"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_notion(self, notion_integration_token, page_ids, database_id=None):

        page_id_list = None
        if page_ids:
            page_id_list = [page_id.strip() for page_id in page_ids.split(",") if page_id.strip()] if page_ids.strip() else None

        db_id = None
        if database_id:
            db_id = database_id.strip() if database_id.strip() else None
        
        if db_id:
            documents = NotionPageReader(integration_token=notion_integration_token).load_data(database_id=db_id)
        else:
            documents = NotionPageReader(integration_token=notion_integration_token).load_data(page_ids=page_id_list)
        
        return (documents,)

```
