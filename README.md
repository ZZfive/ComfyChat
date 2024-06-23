# ComfyChat
Rough LLM Interpreter of ComfyUI

- [简介](#简介)
- [项目发展方向](#项目发展方向)
- [数据](#数据)
- [工作内容](#工作内容)
  - [数据构建](#数据构建)
  - [训练](#训练)
  - [服务部署](#服务部署)
- [工作进度](#工作进度)


## 简介
&emsp;&emsp;因兴趣原因，使用过各类Stable Diffusion模型的GUI项目，如[stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，[Fooocus](https://github.com/lllyasviel/Fooocus)，[ComfyUI](https://github.com/comfyanonymous/ComfyUI)等，ComfyUI节点式的工作方式、活跃的开源社区为其开发的各类功能强大的自定义节点和工作流，均赋予了ComfyUI强大能力，但同时也使其学习成本较高，本项目通过收集社区各中数据构建数据集，使用LLM技术，构建一个降低ComfyUI学习成本的工具。


## 项目发展方向
- 微调：对InternLM2、LLaMA3等LLMs模型微调
- RAG：构建ComfyUI知识库，基于RAG开源框架，改善LLMs回答准确性
- 多模态接入：将ASR、TTS、生图等功能集成
- 工作流训练：ComfyUI工作流本质是表征各个节点相连接的json文档，可能类似于code，尝试通过训练测试LLMs能否理解、构建工作流

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
 - 工作流文件
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
   - 指令微调所需的对话数据构建

 ### 训练
 - InternLM2微调
 - LLaMA3微调

 ### 服务部署
 - 基于微调后LLMs的对话服务部署
 - RAG服务搭建
 - 服务接入TTS
 - 服务接入生图


## 工作进度
 - [x] 对外界面构建
 - [x] 搭建RAG系统
   - [x] 系统搭建
   - [x] 构建的数据转为向量存储
 - [ ] 基于v2数据集训练模型
 - [ ] 优化数据集，构建数据集v2
   - [ ] 搜集一些文档或从视频从提取数据进行构建
   - [x] 已对四个社区集中提供节点文本的项目进行数据构建
 - [x] 基于数据集v1，微调InternLM2-chat-1.8b和InternLM2-chat-7b，模型分别为[zzfive/ComfyChat-InternLM2-1-8b-v1](https://huggingface.co/zzfive/ComfyChat-InternLM2-1-8b-v1)和[zzfive/ComfyChat-InternLM2-7b-v1]()
 - [x] 基于收集的自定义节点项目中的文档，使用deepseek、kimi等LLMs构建了中英文微调数据集，和Aplaca-GPT4数据集混合构建了数据集v1
 - [x] 已收集一批ComfyUI及自定义节点项目文档readme和说明文本


## 致谢
 - 