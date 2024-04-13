# Load Image From Url (mtb)
## Documentation
- Class name: `Load Image From Url (mtb)`
- Category: `mtb/IO`
- Output node: `False`

This node is designed to load an image from a specified URL, facilitating the retrieval and processing of images from the web for further manipulation or analysis.
## Input types
### Required
- **`url`**
    - The URL from which the image is to be loaded. It specifies the web location of the image to be retrieved, playing a crucial role in determining the image that will be processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image retrieved from the specified URL, converted into a tensor format suitable for further image processing tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadImageFromUrl_:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Example.jpg/800px-Example.jpg"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "mtb/IO"

    def load(self, url):
        # get the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        return (pil2tensor(image),)

```
