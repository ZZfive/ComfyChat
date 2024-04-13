# Prune Outputs 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_PruneOutputs`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `True`

This node is designed to selectively delete files based on specified criteria, aiming to manage and optimize storage by removing intermediate or unnecessary output files generated during video processing workflows.
## Input types
### Required
- **`filenames`**
    - Specifies the filenames to be considered for pruning. The selection of files to delete is based on the provided options and the structure of filenames.
    - Comfy dtype: `VHS_FILENAMES`
    - Python dtype: `Tuple[str, List[str]]`
- **`options`**
    - Determines the criteria for file deletion, allowing for the selection between intermediate files, utility files, or all files.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PruneOutputs:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    "options": (["Intermediate", "Intermediate and Utility"],)
                    }
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "prune_outputs"

    def prune_outputs(self, filenames, options):
        assert(len(filenames[1]) <= 3 and len(filenames[1]) >= 2)
        delete_list = []
        if options in ["Intermediate", "Intermediate and Utility", "All"]:
            delete_list += filenames[1][1:-1]
        if options in ["Intermediate and Utility", "All"]:
            delete_list.append(filenames[1][0])
        if options in ["All"]:
            delete_list.append(filenames[1][-1])

        output_dirs = [os.path.abspath("output"), os.path.abspath("temp")]
        for file in delete_list:
            #Check that path is actually an output directory
            if (os.path.commonpath([output_dirs[0], file]) != output_dirs[0]) \
                    and (os.path.commonpath([output_dirs[1], file]) != output_dirs[1]):
                        raise Exception("Tried to prune output from invalid directory: " + file)
            if os.path.exists(file):
                os.remove(file)
        return ()

```
