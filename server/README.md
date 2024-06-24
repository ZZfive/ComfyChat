# 服务说明

- [简介](#简介)
- [服务内容](#服务内容)
- [RAG](#RAG)

## 简介
&emsp;&emsp;本路径下存放服务相关代码，包括模型加载、推理，RAG构建，图片、语音生成等模块；最后通过gradio将所有模块整合对外提供服务。目前服务代码是以“快速展示”为宗旨构建，并没有考虑太多性能方面，后续功能模块完善后再考虑进一步优化。


## 服务内容
 - [x] 简洁界面构建
 - [x] RAG系统搭建
 - [x] LLM本地、远程推理
 - [ ] ASR模块
 - [ ] TTS模块
 - [ ] 生图模块

## RAG
&emsp;&emsp;本服务RAG方案借鉴于[茴香豆](https://github.com/InternLM/HuixiangDou)，从中将文件操作、向量提取及存储、问题查询等子模块抽离出来，基于本项目需求改造。主要相关文件如下：
 - [feature_store.py](feature_store.py)--基本复用茴香豆[feature_store.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/feature_store.py)文件
 - [retriever.py](retriever.py)--基本复用茴香豆[retriever.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/retriever.py)文件
 - [llm_infer.py](llm_infer.py)--主要参考以下两个文件构建复合LLM推理类，可有效基于本地模型或调用远端接口进行推理
   - [llm_client.py](source/llm_client.py)--茴香豆中客户端调用的llm[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_client.py)，本文件进行了适当注释，可参考
   - [llm_server_hybrid.py](source/llm_server_hybrid.py)--茴香豆中llm的服务端[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_server_hybrid.py)，本文件进行了适当注释，可参考

&emsp;&emsp;茴香豆具有拒答功能，考虑到ComfyUI社区极其活跃，当前收集资料必然不全，故保留了拒答功能，通过其判断当前数据未覆盖的问题，直接用LLM推理生成答案。基于茴香豆构建RAG的过程极大加深对RAG架构了解，是很好的学习经验，感兴趣的朋友可以从茴香豆中学习。
