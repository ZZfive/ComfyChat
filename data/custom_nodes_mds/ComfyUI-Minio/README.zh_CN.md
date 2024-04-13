# Comfyui-Minio
本插件主要是基于minio，实现从minio里面读取、保存图片，方便做多个机器的扩展和连接

# 切换语言

- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# 节点

|名称                         |描述                                                                               |
|----------------------------|---------------------------------------------------------------------------------- |
|Set Minio Config            |minio的初始化，在使用本插件前，请先使用本节点进行配置                                  |
|Load Image From Minio       |从minio读取图片（注意：在使用前，请先使用Set Minio Config进行初始化）                  |
|Save Image To Minio         |将图片保存到minio，支持多图的保存（注意：在使用前，请先使用Set Minio Config进行初始化）  |

# 需要安装的依赖

本插件需要安装minio的python sdk
```
pip install -r requirements.txt
```

# 初始化步骤
在正式使用前，请先根据下面的步骤进行初始化

## 1. 添加节点Set Minio Config
![steps 1](../docs/steps-image-1.png)


## 2. 输入你的minio配置信息, 并且运行该插件
![steps 2](../docs/steps-image-2.png)

注意：如果想要显示json信息，请安装插件[Comfyui-Toolbox](https://github.com/zcfrank1st/Comfyui-Toolbox)

## 3. 如果minio配置正确，且可以正常连接，则会在output/minio_config.json

```
ComfyUI
    output
        minio_config.json
```

## 4. 接下来你就可以正常使用其它两个节点了
![Comfyui-Minio-workflow](../docs/workflow.png)