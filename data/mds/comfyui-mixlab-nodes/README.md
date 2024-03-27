> 适配了最新版comfyui的py3.11 ，torch 2.1.2+cu121

> [Mixlab nodes discord](https://discord.gg/cXs9vZSqeK)

#### 
[comfyui-Image-reward](https://github.com/shadowcz007/comfyui-Image-reward)

[comfyui-ultralytics-yolo](https://github.com/shadowcz007/comfyui-ultralytics-yolo)

[comfyui-moondream](https://github.com/shadowcz007/comfyui-moondream)

[comfyui-CLIPSeg](https://github.com/shadowcz007/comfyui-CLIPSeg)


## 🚀🚗🚚🏃 Workflow-to-APP 
- 新增AppInfo节点，可以通过简单的配置，把workflow转变为一个Web APP。
- 支持多个web app 切换
- 发布为app的workflow，可以在右键里再次编辑了
- web app可以设置分类，在comfyui右键菜单可以编辑更新web app


- Support multiple web app switching.
- Add the AppInfo node, which allows you to transform the workflow into a web app by simple configuration.
- The workflow, which is now released as an app, can also be edited again by right-clicking.
- The web app can be configured with categories, and the web app can be edited and updated in the right-click menu of ComfyUI.


![](./assets/0-m-app.png)

![](./assets/appinfo-readme.png)

![](./assets/appinfo-2.png)

Example:
- workflow
![APP info](./workflow/appinfo-workflow.svg)
[text-to-image](./workflow/Text-to-Image-app.json)

APP-JSON:
- [text-to-image](./example/Text-to-Image_3.json)
- [image-to-image](./example/Image-to-Image_2.json)
- text-to-text

> 暂时支持 9 种节点作为界面上的输入节点：Load Image、VHS_LoadVideo、CLIPTextEncode、PromptSlide、TextInput_、Color、FloatSlider、IntNumber、CheckpointLoaderSimple、LoraLoader

> 输出节点：PreviewImage 、SaveImage、ShowTextForGPT、VHS_VideoCombine、PromptImage

> seed统一输入控件，支持：SamplerCustom、KSampler

> 配套[ps插件](https://github.com/shadowcz007/comfyui-ps-plugin)

> 如果遇到上传图片不成功，请检查下：局域网或者是云服务，请使用https，端口8189这个服务（ 感谢 @Damien 反馈问题）

> If you encounter difficulties in uploading images, please check the following: for local network or cloud services, please use HTTPS and the service on port 8189. (Thanks to @Damien for reporting the issue.)



## 🏃🚗🚚🚀  Real-time Design
> ScreenShareNode & FloatingVideoNode. Now comfyui supports capturing screen pixel streams from any software and can be used for LCM-Lora integration. Let's get started with implementation and design! 💻🌐

![screenshare](./assets/screenshare.png)

https://github.com/shadowcz007/comfyui-mixlab-nodes/assets/12645064/e7e77f90-e43e-410a-ab3a-1952b7b4e7da


<!-- [ScreenShareNode](./workflow/2-screeshare.json) -->
[ScreenShareNode & FloatingVideoNode](./workflow/3-FloatVideo-workflow.json)

!! Please use the address with HTTPS (https://127.0.0.1).


### SpeechRecognition & SpeechSynthesis
![f](./assets/audio-workflow.svg)

[Voice + Real-time Face Swap Workflow](./workflow/语音+实时换脸workflow.json)

### GPT
> Support for calling multiple GPTs.ChatGPT、ChatGLM3 、ChatGLM4 , Some code provided by rui. If you are using OpenAI's service, fill in https://api.openai.com/v1 . If you are using a local LLM service, fill in http://127.0.0.1:xxxx/v1 .  Azure OpenAI:https://xxxx.openai.azure.com 

![gpt-workflow.svg](./assets/gpt-workflow.svg)

[workflow-5](./workflow/5-gpt-workflow.json)


## Prompt
> PromptSlide
![](./assets/prompt_weight.png)

<!-- ![](./workflow/promptslide-appinfo-workflow.svg) -->

> randomPrompt

![randomPrompt](./assets/randomPrompt.png)

> ClipInterrogator

[add clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator)

> PromptImage & PromptSimplification,Assist in simplifying prompt words, comparing images and prompt word nodes.

> ChinesePrompt && PromptGenerate，中文prompt节点，直接用中文书写你的prompt

![](./assets/ChinesePrompt_workflow.svg)


### Layers
> A new layer class node has been added, allowing you to separate the image into layers. After merging the images, you can input the controlnet for further processing.

![layers](./assets/layers-workflow.svg)

![poster](./assets/poster-workflow.svg)


### 3D
![](./assets/3dimage.png)
[workflow](./workflow/3D-workflow.json)


### LoadImagesFromLocal
> Monitor changes to images in a local folder, and trigger real-time execution of workflows, supporting common image formats, especially PSD format, in conjunction with Photoshop. 

![watch](./assets/4-loadfromlocal-watcher-workflow.svg)

[workflow-4](./workflow/4-loadfromlocal-watcher-workflow.json)

### LoadImagesFromURL
> Conveniently load images from a fixed address on the internet to ensure that default images in the workflow can be executed.


## Style
> Apply VisualStyle Prompting , Modified from [ComfyUI_VisualStylePrompting](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting) 

![](./assets/VisualStylePrompting.png)

> StyleAligned , Modified from [style_aligned_comfy](https://github.com/brianfitzgerald/style_aligned_comfy) 


## Utils
> The Color node provides a color picker for easy color selection, the Font node offers built-in font selection for use with TextImage to generate text images, and the DynamicDelayByText node allows delayed execution based on the length of the input text.

- [添加了DynamicDelayByText功能，可以根据输入文本的长度进行延迟执行。](./workflow/audio-chatgpt-workflow.json)

- [Added DynamicDelayByText, enabling delayed execution based on input text length.](./workflow/audio-chatgpt-workflow.json)

- [使用CkptNames 对比不同的模型效果](./workflow/ckpts-image-workflow.json)

- [CkptNames compare the effects of different models.](./workflow/ckpts-image-workflow.json)



## Other Nodes

![main](./assets/all-workflow.svg)
![main2](./assets/detect-face-all.png)

[workflow-1](./workflow/1-workflow.json)



> TransparentImage

![TransparentImage](./assets/TransparentImage.png)


> FeatheredMask、SmoothMask

Add edges to an image.

![FeatheredMask](./assets/FlVou_Y6kaGWYoEj1Tn0aTd4AjMI.jpg)


> LaMaInpainting

from [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)


> rembgNode

"briarmbg","u2net","u2netp","u2net_human_seg","u2net_cloth_seg","silueta","isnet-general-use","isnet-anime"

*** briarmbg ***  model was developed by BRlA Al and can be used as an open-source model for non-commercial purposes


### Improvement 

- Add "help" option to the context menu for each node.
- Add "Nodes Map" option to the global context menu.

An improvement has been made to directly redirect to GitHub to search for missing nodes when loading the graph.

![help](./assets/help.png)

![node-not-found](./assets/node-not-found.png)


### Models

[Download rembg Models](https://github.com/danielgatis/rembg/tree/main#Models),move to:models/rembg

[Download lama](https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt), move to : models/lama

[Download Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base), move to : models/clip_interrogator/Salesforce/blip-image-captioning-base

[Download succinctly/text2image-prompt-generator](https://huggingface.co/succinctly/text2image-prompt-generator/tree/main),move to:prompt_generator/text2image-prompt-generator

[Download Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en/tree/main),move to:prompt_generator/opus-mt-zh-en

## Installation

manually install, simply clone the repo into the custom_nodes directory with this command:

```
cd ComfyUI/custom_nodes

git clone https://github.com/shadowcz007/comfyui-mixlab-nodes.git

```

Install the requirements:

run directly:
```
cd ComfyUI/custom_nodes/comfyui-mixlab-nodes
install.bat
```

or install the requirements using:
```
../../../python_embeded/python.exe -s -m pip install -r requirements.txt
```

If you are using a venv, make sure you have it activated before installation and use:
```
pip3 install -r requirements.txt
```





#### Chinese community
访问 [www.mixcomfy.com](https://www.mixcomfy.com)，获得更多内测功能，关注微信公众号：Mixlab无界社区




#### 
File / LoadImagesFromPath SaveImageToLocal LoadImagesFromURL




#### discussions:
[discussions](https://github.com/shadowcz007/comfyui-mixlab-nodes/discussions)


<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=shadowcz007/comfyui-mixlab-nodes&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=shadowcz007/comfyui-mixlab-nodes&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=shadowcz007/comfyui-mixlab-nodes&type=Date"
  />
</picture>

