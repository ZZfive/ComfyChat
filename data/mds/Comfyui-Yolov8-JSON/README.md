# Comfyui-Yolov8-JSON
This node is mainly based on the Yolov8 model for object detection, and it outputs related images, masks, and JSON information.

![image](https://github.com/Alysondao/Comfyui-Yolov8-JSON/blob/main/docs/workflow.png)
![image](https://github.com/Alysondao/Comfyui-Yolov8-JSON/blob/main/docs/workflow1.png)

# README
- [English](README.md)
- [简体中文](readme/README.zh_CN.md)

# NODES
|name                         |description                                                     |
|-----------------------------|----------------------------------------------------------------|
|Load Yolov8 Model            |Default, select the Yolov8 model, supports automatic download   |
|Load Yolov8 Model From Path  |Load the model from the specified path                          |
|Apply Yolov8 Model           |Apply Yolov8 detection model                                    |
|Apply Yolov8 Model Seg       |Apply Yolov8 segmentation model                                 |

# INSTALL
If you need to display JSON formatted data or save it, you need to install the [Comfyui-Toolbox](https://github.com/zcfrank1st/Comfyui-Toolbox) node in advance.

This node calls the official Python package, and you also need to install the following dependencies:

```
pip install -r requirements.txt
```

# MODEL
This node supports automatic model downloads.
You can manually download the models to the specified folder `models/yolov8` from [Yolov8](https://github.com/ultralytics/ultralytics) as shown below:


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
 
