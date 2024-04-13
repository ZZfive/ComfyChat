# LLM Simple Directory Reader
## Documentation
- Class name: `LLMDirectoryReader`
- Category: `SALT/Llama-Index/Readers`
- Output node: `False`

The LLMDirectoryReader node is designed to read and process documents from a specified directory. It supports filtering based on file extensions and exclusion patterns, and can operate recursively. This node is essential for ingesting and preparing local document collections for further processing or analysis.
## Input types
### Required
- **`input_directory`**
    - Specifies the subdirectory within the input directory from which to read documents. It is crucial for targeting the specific location of the documents to be processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`recursive`**
    - Determines whether the directory reading should be recursive, allowing for the processing of documents in subdirectories.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`required_ext_list`**
    - A comma-separated list of file extensions to include in the reading process, enabling the filtering of documents by type.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_glob_list`**
    - A comma-separated list of glob patterns to exclude from the reading process, allowing for more refined control over which documents are processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_documents`**
    - Comfy dtype: `LLM_DOCUMENTS`
    - The documents read from the specified directory, filtered according to the provided criteria.
    - Python dtype: `Tuple[List[str]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMDirectoryReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": False}),
                "required_ext_list": ("STRING", {"default": ".json, .txt, .html"}),
                "exclude_glob_list": ("STRING", {"default": ".sqlite, .zip"}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_directory"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_directory(self, input_directory, recursive=False, required_ext_list=None, exclude_glob_list=None):
        full_path = os.path.join(folder_paths.get_input_directory(), input_directory.strip())

        input_dir = full_path if os.path.isdir(full_path) else None
        if not input_dir:
            raise ValueError("The provided subdirectory does not exist.")
        
        if required_ext_list.strip():
            required_exts = [ext.strip() for ext in required_ext_list.split(",") if ext.strip()]
        else:
            required_exts = None

        if exclude_glob_list.strip():
            exclude = [pattern.strip() for pattern in exclude_glob_list.split(",") if pattern.strip()]
        else:
            exclude = None

        print("Excluding: ", exclude)
        print("Required Extensions: ", required_exts)
        
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            exclude_hidden=True,
            recursive=recursive,
            required_exts=required_exts,
            exclude=exclude
        )

        documents = reader.load_data()
        if not documents:
            raise ValueError("No documents found in the specified directory.")

        return (documents,)

```
