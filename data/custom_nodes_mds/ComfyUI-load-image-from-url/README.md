A simple node to load image from local path or http url.

You can find this node from "image" category.

![](https://i.postimg.cc/XY0CLx3Q/image.png)

```json
{
  "1": {
    "inputs": {
      "url_or_path": "https://i.postimg.cc/TYbdk63X/test.png"
    },
    "class_type": "LoadImageFromUrlOrPath",
    "_meta": {
      "title": "LoadImageFromUrlOrPath"
    }
  },
  "2": {
    "inputs": {
      "images": [
        "1",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  }
}
```
