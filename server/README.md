# 服务说明

- [简介](#简介)
- [构建内容](#构建内容)
- [界面构建](#界面构建)
- [RAG](#RAG)
- [LLM推理](#LLM推理)
- [ASR模块](#ASR模块)
- [TTS模块](#TTS模块)
- [生图模块](#生图模块)

## 简介
&emsp;&emsp;本路径下存放服务相关代码，包括模型加载、推理，RAG构建，图片、语音生成等模块；最后通过gradio将所有模块整合对外提供服务。目前服务代码是以“快速展示”为宗旨构建，并没有考虑太多性能方面，后续功能模块完善后再考虑进一步优化。


## 构建内容
 - [x] 简洁界面构建
 - [x] RAG系统搭建
 - [x] LLM推理
 - [x] ASR模块
 - [x] TTS模块
 - [ ] 生图模块
 
 
## 界面构建
&emsp;&emsp;本服务使用gradio快速搭建服务界面，主要使用Chatbot、Audio等组件搭建，使用者可极易复现、启动和使用。注意，基于[Chatbot.select](https://www.gradio.app/docs/gradio/chatbot#event-listeners)事件监听实现了对Chatbot中LLM生成的文本点击动作触发TTS翻译，可参考[chatbot_test.py](source/chatbot_test.py)分析实现原理，，而demo中的具体实现参考[此处](demo.py#165)

## RAG
&emsp;&emsp;本服务RAG方案借鉴于[茴香豆](https://github.com/InternLM/HuixiangDou)，从中将文件操作、向量提取及存储、问题查询等子模块抽离出来，基于本项目需求改造。主要相关文件如下：
 - [feature_store.py](feature_store.py)--基本复用茴香豆[feature_store.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/feature_store.py)文件
 - [retriever.py](retriever.py)--基本复用茴香豆[retriever.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/retriever.py)文件


&emsp;&emsp;茴香豆具有拒答功能，考虑到ComfyUI社区极其活跃，当前收集资料必然不全，故保留了拒答功能，通过其判断当前数据未覆盖的问题，直接用LLM推理生成答案。基于茴香豆构建RAG的过程极大加深对RAG架构了解，是很好的学习经验，感兴趣的朋友可以从茴香豆中学习。


## LLM推理
&emsp;&emsp;LLM推理部分也主要借鉴茴香豆，llm_infer.py中将调用远程LLM接口和本地LLM模型推理集成在一起，可通过config.ini文件进行配置
 - [llm_infer.py](llm_infer.py)--主要参考以下两个文件构建复合LLM推理类，可有效基于本地模型或调用远端接口进行推理
   - [llm_client.py](source/llm_client.py)--茴香豆中客户端调用的llm[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_client.py)，本文件进行了适当注释，可参考
   - [llm_server_hybrid.py](source/llm_server_hybrid.py)--茴香豆中llm的服务端[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_server_hybrid.py)，本文件进行了适当注释，可参考

## ASR模块
&emsp;&emsp;本模块能将语音转换为文本，使用了基于[openai whisper](https://github.com/openai/whisper)衍生的两种方案，分别是[whispercpp](https://github.com/AIWintermuteAI/whispercpp)和[whisperX](https://github.com/m-bain/whisperX)，两种方案表现区别不大，whispercpp底层基于[whisper.cpp](https://github.com/ggerganov/whisper.cpp)，whisperX底层基于[faster-whisper](https://github.com/guillaumekln/faster-whisper)，均是用C/C++执行计算；但从安装快捷性上，推荐使用whisperX

## TTS模块
&emsp;&emsp;本模块能将LLM推理生成的文本转为语音，测试了[Chattts](https://github.com/2noise/ChatTTS)和[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，两者都是今年很火的TTS模型。Chattts效果测试下来，应该算是一般，并且生成的语音中会出现输入文本中不存在的词汇；GPT-SoVITS的推理功能抽取还未完全完成，还要写小问题，故目前demo暂时使用的是Chatts生成语音。


## 生图模块
&emsp;&emsp;