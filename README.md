# ComfyChat
Rough LLM Interpreter of ComfyUI

- [简介](#简介)
- [架构图](#架构图)
- [部署](#部署)
- [项目发展方向](#项目发展方向)
- [数据](#数据)
- [工作内容](#工作内容)
  - [数据构建](#数据构建)
  - [训练](#训练)
  - [服务部署](#服务部署)
- [工作进度](#工作进度)


## 简介
&emsp;&emsp;**本项目依托书生·浦语大模型实战营活动，以期基于LLMs能力，针对ComfyUI构建一个粗糙但相对完善的解释器；帮助对ComfyUI、Stable Diffusion感兴趣的朋友快速上手是本项目的宗旨。非常感谢上海人工智能实验室提供此次机会，同时也非常感谢本项目涉及到的开源项目和个人！！！**

&emsp;&emsp;因兴趣原因，使用过各类Stable Diffusion模型的GUI项目，如[stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，[Fooocus](https://github.com/lllyasviel/Fooocus)，[ComfyUI](https://github.com/comfyanonymous/ComfyUI)等，ComfyUI节点式的工作方式、活跃的开源社区为其开发的各类功能强大的自定义节点和工作流，均赋予了ComfyUI强大能力，但同时也使其学习成本较高，本项目通过收集社区各中数据构建数据集，使用LLM技术，构建一个降低ComfyUI学习成本的工具。

## 架构图
![enter image description here](assets/ComfyChat.png?raw=true)

## 部署
&emsp;&emsp;本项目服务主体代码在[server](server)路径下，[demo.py](server/demo.py)是服务入口，运行起来不太复杂，可按以下步骤操作：
1. 推荐使用conda或miniconda新建一个python310的虚拟环境并激活，即'conda create -n ComfyChat python=3.10'、'conda activate ComfyChat'
2. [requirements.txt](server/requirements.txt)中已指定主要依赖库和版本，在ComfyChat中安装即可。不推荐直接使用'pip install -r requirements.txt'安装，因为依赖库较多，一个库安装失败会导致所有库安装失败。requirements.txt中各库版本是项目开发环境中使用版本，应该不会出现版本冲突，但不保证。
3. 拉取本项目代码：git clone --recurse-submodules https://github.com/ZZfive/ComfyChat.git
4. 为方便模型管理，服务依赖模型可统一存放在[weights](weights)路径下，其中主要存放Whisperx和GPT-SoVITS依赖模型，LLM和Embedding的模型也能存放在此路径下，只需在[config.ini](server/config.ini)正确配置即可。Whisperx依赖模型在demo.py运行时会自动下载至weights中，GPT-SoVITS依赖模型则需手动下载。先在weights路径下创建GPT-SoVITS，将相关模型存放在weights/GPT-SoVITS中；此路径接口图如下所示，其中weights/GPT_SoVITS/pretrained_models路径下存放的是GPT-SoVITS项目提供的预训练模型，可在该项目中找到下载地址；而剩下的人物角色模型是开源社区贡献的，可在[此处](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/nwnaga50cazb2v93)下载
![enter image description here](assets/weights_gptsovits.png?raw=true)
5. 若想开启comfyui，需要先准备一些必要模型，如Stable-diffusion、Lora、VAE、embeddings和controlnet等，可以将此类模型都存放在[weights](weights)路径下，然后修改[visual_genrations/comfyui/extra_model_paths.yaml.example](/root/code/ComfyChat/visual_genrations/comfyui/extra_model_paths.yaml.example)文件；先将文件名中的.example删掉，改为extra_model_paths.yaml，然后再按以下截图中设置模型路径。模型准备好后，将config.ini中comfyui配置中的enable设置为1
![enter image description here](assets/comfyui_models.png?raw=true)
6. 执行完上述操作后，正确设置[config.ini](server/config.ini)，'python demo.py'就能成功运行服务

## 项目发展方向
- 已进行
   - [x] 微调：对InternLM2、LLaMA3等LLMs模型微调
   - [x] RAG：构建ComfyUI知识库，基于RAG开源框架，改善LLMs回答准确性
   - [x] 多模态接入：将ASR、TTS、生图等功能集成
- 待开展
   - [ ] 工作流训练：ComfyUI工作流本质是表征各个节点相连接的json文档，可能类似于code，尝试通过训练测试LLMs能否理解、构建工作流

 ## 数据

数据源
 - 各默认节点的解释说明--comfyui项目文档及网上数据，社区的节点说明文档
   - [https://github.com/BlenderNeko/ComfyUI-docs](https://github.com/BlenderNeko/ComfyUI-docs)
 - 自定义节点的解释说明--各类自定义节点项目中的说明文档
   - 从[
ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)收集的自定义节点项目列表文件中构建批量下载
   - [https://github.com/6174/comflowy](https://github.com/6174/comflowy)
   - [https://github.com/get-salt-AI/SaltAI-Web-Docs](https://github.com/get-salt-AI/SaltAI-Web-Docs)
   - [https://github.com/CavinHuang/comfyui-nodes-docs](https://github.com/CavinHuang/comfyui-nodes-docs)
 - 工作流文件--仍未实际开展
   - [c站](https://civitai.com/)
   - github上的个人汇总
   - [https://comfyworkflows.com](https://comfyworkflows.com)
   - [https://openart.ai/workflows](https://openart.ai/workflows)

## 工作内容

### 数据构建
 - 数据收集
   - comfyui说明数据爬取
     - [x] 社区说明[文档](https://blenderneko.github.io/ComfyUI-docs/#further-support)信息收集
     - [x] comfyui manager中的节点列表信息收集
     - [ ] 网络上关于comfyui的信息收集--ing
     - [x] 部署一个视频、音频转文本模型，收集相关文本--whisperx
   - comfyui工作流及对应文本说明数据爬取
     - [ ] 待开展
   - 去重？
   - 安全过滤？

 - 数据集构建
   - 指令微调所需的对话数据构建：已构建v1、v2数据，详细数据跳转至[此处](data/message_jsons/README.md)

 ### 训练
 - [x] InternLM2微调
 - [ ] LLaMA3微调

 ### 服务部署
 - [x] 基于微调后LLMs的对话服务部署
 - [x] RAG服务搭建
 - [x] 服务接入TTS
 - [x] 服务接入生图


## 工作进度
 - [ ] 优化数据集，构建数据集v3
   - [ ] 搜集一些文档或从视频从提取数据进行构建
 - [x] 对外界面构建
   - [x] 推理模块--基于gradio
   - [x] TTS模块--基于[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
   - [x] ASR模块--基于[whisperX](https://github.com/m-bain/whisperX)
   - [x] 服务接入生图--基于[ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 - [x] 搭建RAG系统--基于[茴香豆](https://github.com/InternLM/HuixiangDou)
   - [x] 系统搭建
   - [x] 构建的数据转为向量存储
 - [x] 基于v2数据集训练模型
   - [x] 已对四个社区集中提供节点文本的项目进行数据构建
 - [x] 基于数据集v1，微调InternLM2-chat-1.8b和InternLM2-chat-7b，模型分别为[zzfive/ComfyChat-InternLM2-1-8b-v1](https://huggingface.co/zzfive/ComfyChat-InternLM2-1-8b-v1)和[zzfive/ComfyChat-InternLM2-7b-v1](https://huggingface.co/zzfive/ComfyChat-InternLM2-7b-v1)
 - [x] 基于收集的自定义节点项目中的文档，使用deepseek、kimi等LLMs构建了中英文微调数据集，和Aplaca-GPT4数据集混合构建了数据集v1
 - [x] 已收集一批ComfyUI及自定义节点项目文档readme和说明文本


## 致谢
特别感谢以下团队、项目或人员
 - 上海人工智能实验室和书生·浦语团队
 - [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 - [https://github.com/BlenderNeko/ComfyUI-docs](https://github.com/BlenderNeko/ComfyUI-docs)
 - [https://github.com/6174/comflowy](https://github.com/6174/comflowy)
 - [https://github.com/get-salt-AI/SaltAI-Web-Docs](https://github.com/get-salt-AI/SaltAI-Web-Docs)
 - [https://github.com/CavinHuang/comfyui-nodes-docs](https://github.com/CavinHuang/comfyui-nodes-docs)
 - [茴香豆](https://github.com/InternLM/HuixiangDou)
 - [whisperX](https://github.com/m-bain/whisperX)
 - [whispercpp](https://github.com/AIWintermuteAI/whispercpp)
 - [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
 - [Chattts](https://github.com/2noise/ChatTTS)
 - [旭_1994](https://blog.csdn.net/qq_38944169?type=blog)的[CSDN博客](https://blog.csdn.net/qq_38944169/article/details/139245317)