# Comfyui-Minio
This plugin is mainly based on Minio, implementing the ability to read images from Minio, save images, facilitating expansion and connection across multiple machines.

# Language

- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# Nodes

|Name                         |Description                                            |
|-----------------------------|-------------------------------------------------------|
|Set Minio Config             |Initialization of Minio.                               |
|Load Image From Minio        |Read images from Minio.                                |
|Save Image To Minio          |Save images to Minio, supports saving multiple images. |

# Install

This plugin requires installation of Minio's Python SDK

```
pip install -r requirements.txt
```

# Note
Before using, please initialize according to the steps below

## 1. Add node:Set Minio Config

![steps 1](./docs/steps-image-1.png)


## 2. Setting
Enter your Minio configuration information and run the plugin

![steps 2](./docs/steps-image-2.png)

Note: If you want to display JSON information, please install the plugin [Comfyui-Toolbox](https://github.com/zcfrank1st/Comfyui-Toolbox)

## 3. Check
If the Minio configuration is correct and can be connected successfully, a file will be created in the output directory

```
ComfyUI
    output
        minio_config.json
```

## 4. End
Now you can proceed to use the other two nodes as normal
![Comfyui-Minio-workflow](./docs/workflow.png)