# G'MIC in ComfyUI

![](./screenshot--comfyui_gmic.png)



## Features

* Wonderful image-processing suite (effects, filtering, geometric manipulation, image data, color.)
* Friendly for beginners and power users.
* On going [documented](https://github.com/gemell1/ComfyUI_GMIC/wiki) workflows.

##  Quick Start

- Install custom node with: `git clone https://github.com/gemell1/ComfyUI_GMIC`
  or copy [comfy_gmic.py](./comfy_gmic.py) file to custom nodes folder.
- Download [G'MIC Command-line interface (CLI)](https://gmic.eu/download.html) and [G'MIC-Qt stand-alone interface](https://gmic.eu/download.html).
- Put executable path in your PATH. ([how to](https://windowsloop.com/how-to-add-to-windows-path/))
- Drag and drop this [workflow](./workflow.json) to ComfyUi.  

```py
### Changelog ----------
- 'March' Breaking changes: please recreate your workflow! 🙏
    @ G_MIC Qt Node
    @ Batch Images
- "Feb" Better cross platform support.
- "Nov" First upload.
```

---

Tested on Windows 10, ComfyUI [March 09](https://github.com/comfyanonymous/ComfyUI/tree/a9ee9589b72aa0e2931f1c0705524c56adaee26d), G'MIC (Cli and Qt) 3.3.4

---

[Other Image Processing ComfyUI custom nodes](https://github.com/gemell1/ComfyUI_GMIC/wiki)
