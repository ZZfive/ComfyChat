# Comfyui-Yolov8-JSON
本节点主要是基于yolov8模型，进行物体的检测，并且输出相关的图片、蒙版和JSON信息。

![image](https://github.com/Alysondao/Comfyui-Yolov8-JSON/blob/main/docs/workflow.png)
![image](https://github.com/Alysondao/Comfyui-Yolov8-JSON/blob/main/docs/workflow1.png)

# 切换语言
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# 节点
|名称                          |描述                             |
|------------------------------|--------------------------------|
|Load Yolov8 Model            |默认，选择yolov8模型，支持自动下载  |
|Load Yolov8 Model From Path  |从指定地址读取模型                 |
|Apply Yolov8 Model           |yolov8检测模型推理                 |
|Apply Yolov8 Model Seg       |yolov8分割模型推理                 |

# 需要安装的依赖
如果你需要显示JSON格式的数据，或者保存，你需要提前安装[Comfyui-Toolbox 节点](https://github.com/zcfrank1st/Comfyui-Toolbox)

本节点调用的是官方提供的python包,你还需要安装下面的依赖

```
pip install -r requirements.txt
```

# 模型
本节点支持自动下载模型
你也可以自己去[Yolov8](https://github.com/ultralytics/ultralytics)手动下载模型到指定文件夹`models/yolov8` 
如下：

```
ComfyUI
    models
        yolov8
            yolov8l.pt
            yolov8s.pt
            yolov8n.pt
            yolov8m.pt
            yolov8x.pt
            yolov8l-seg.pt
            yolov8s-seg.pt
            yolov8n-seg.pt
            yolov8m-seg.pt
            yolov8x-seg.pt
```
