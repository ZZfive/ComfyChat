# ComfyUI_ImageToText

## 功能简述

1. 这是ComfyUI的节点，可以在ComfyUI中使用
2. 功能是将图片以自然语言描述出来.
3. 提供了按照文件夹批量处理图片的脚本：[BatchImageToText.py](BatchImageToText.py)，打标效果评测：https://mp.weixin.qq.com/s/8R_mPVmYyw4NX5zlaMCflw

## 使用图例

![demo.png](image%2Fdemo.png)

A ginger cat with white paws and chest is sitting on a snowy field, facing the camera with its head tilted slightly to the left. The cat's fur is a mix of white and orange, and its eyes are a striking blue. The background features a snowy field with trees in the distance, and the sun is shining brightly, casting a warm glow on the scene.

翻译为中文：一只 ginger 猫，有白毛和胸口，正坐在一个雪地里，面朝相机，头部稍微向左倾斜。猫的毛色是白色和橙色混合的，眼睛是蓝色。背景上，有雪地，有远处的树木，太阳正在晒出暖暖的阳光。

测试报告：https://mp.weixin.qq.com/s/8R_mPVmYyw4NX5zlaMCflw

## 工作流举例

SDXL模型下载地址(欢迎点赞点关注): https://www.liblib.art/modelinfo/5913fb0765ce4a4ba210cb1c898df276
工作流文件（直接拖拽到ComfyUI的页面里即可）: [ComfyUI-ImageToText.json](ComfyUI-ImageToText.json)

## 使用了的模型

https://huggingface.co/vikhyatk/moondream2

## For More

### ComfyUI
1. 将中文翻译英文：https://github.com/SoftMeng/ComfyUI-FanYi
2. 从图片提取自然语言：https://github.com/SoftMeng/ComfyUI_ImageToText
3. 随机生成提示词：https://github.com/SoftMeng/ComfyUI-Prompt
4. 通过HTML模版制作AI海报：https://github.com/SoftMeng/ComfyUI_Mexx_Poster
5. 通过图片模版制作AI海报：https://github.com/SoftMeng/ComfyUI_Mexx_Image_Mask
6. Java工程调用ComfyUI生成AI图片（含全自动图片馆）：https://github.com/SoftMeng/comfy-flow-api
### Stable Diffusion WebUI
1. 随机生成提示词：https://github.com/SoftMeng/stable-diffusion-prompt-pai
### Fooocus
1. 汉化：https://github.com/SoftMeng/Fooocus-zh
### 其他
2. 视频会议：https://github.com/SoftMeng/vue-webrtc