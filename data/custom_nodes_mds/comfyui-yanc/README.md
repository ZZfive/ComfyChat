# comfyui-yanc

Yet Another Node Collection, a repository of simple nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

This repository was created to alleviate a few of the problems I experienced using other ComfyUI custom node repositories:
1. I frequently wanted only one custom node out of several in a repository.
2. Some custom nodes were more complex than necessary, making them and dependent workflows inaccessible or otherwise difficult to use after a ComfyUI update.
3. Some custom nodes were updated such that dependent workflows became inaccessible or otherwise difficult to use afterwards.

This repository eases the addition or removal of custom nodes to itself.
1. Custom node scripts can follow the format used by [example_node.py.example](https://github.com/comfyanonymous/ComfyUI/blob/master/custom_nodes/example_node.py.example).
2. Add or remove nodes by adding or removing scripts to or from the [nodes](nodes) directory.      
    * Credit to [pythongosssss](https://github.com/pythongosssss) for [\_\_init\_\_.py](__init__.py), the script which discovers and imports nodes.

![Preview Image of Nodes included in comfyui-yanc](nodes.png)
