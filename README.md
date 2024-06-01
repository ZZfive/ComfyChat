# ComfyChat
Rough LLM Interpreter of ComfyUI

- [简介](#简介)
- [数据](#数据)
- [暂定方向](#暂定方向)
- [工作内容](#工作内容)
- [工作进度](#工作进度)

## 简介
&emsp;&emsp;因兴趣原因，使用过各类Stable Diffusion模型的GUI项目，如[stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，[Fooocus](https://github.com/lllyasviel/Fooocus)，[ComfyUI](https://github.com/comfyanonymous/ComfyUI)等，ComfyUI节点式的工作方式、活跃的开源社区为其开发的各类功能强大的自定义节点和工作流，均赋予了ComfyUI强大能力，但同时也使其学习成本较高，为了提高ComfyUI的使用效率，想到能否通过微调LLM或者RAG的方式构建一个ComfyUI的解释器

## 数据

暂定数据源
 - 各默认节点的解释说明--comfyui项目文档及网上数据，社区的节点说明文档
 - 自定义节点的解释说明--各类自定义节点项目中的说明文档，可以从[
ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)收集的自定义节点项目列表文件中构建批量下载
 - 工作流文件
  - [c站](https://civitai.com/)
  - github上的个人汇总
  - workflows网站


## 暂定方向
- 微调InternLM使其学习到ComfyUI的知识？
- 通过RAG的方式构建知识库？
- 收集大量workflow的json文档，以微调coding能力的方式微调InternLM，看能否使InternLM具有理解、分析和构建工作流的能力
- 后续有能力可能往多模态方向发展

训练待确认问题
 - 当前需求使用微调好还是RAG方案更好？
 - 微调的话，是使用SFT还是指令微调？
 - 想做到同时对英文和中文都有回答能力，是只需用一个语言的数据集训练就行，还是要分别构建英文和中文数据集？
 - 如何评估性能？？？

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
   - 网页
     1. 人工搜集comfyui的网页，然后使用Trafilatura进行文本抽取
     2. 看能不能基于什么已有项目，从一个初始url，递归收取内部含有的url文本
   - 去重？
   - 安全过滤？

 - 数据集构建
   - 可能的sft数据构建
   - 指令微调所需的对话数据构建
   - 搭建一个RAG基于搜集的原始数据构建问答数据对？
 - 问答对数据
   - 星火api自动脚本
   - pi模型人工对话合成
   - gpt3.5人工对话
   - 使用收集的原始数据和llm（kimi chat等）构建训练可用数据
 
 - 训练
   - 微调

## 工作进度
 - [x] 已收集一批ComfyUI及自定义节点项目文档readme和说明文本
 - [ ] 已收集了一批相关网页url且下载了文本信息；还需继续补充
 - [ ] workflow收集还未开始
