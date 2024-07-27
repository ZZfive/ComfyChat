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
 - [x] 生图模块
 
 
## 界面构建
&emsp;&emsp;本服务使用gradio快速搭建服务界面，主要使用Chatbot、Audio等组件搭建，使用者可极易复现、启动和使用。注意，基于[Chatbot.select](https://www.gradio.app/docs/gradio/chatbot#event-listeners)事件监听实现了对Chatbot中LLM生成的文本点击动作触发TTS翻译，可参考[chatbot_test.py](source/chatbot_test.py)分析实现原理，而demo中的具体实现参考[此处](demo.py#185)

## RAG
&emsp;&emsp;本服务RAG方案借鉴于[茴香豆](https://github.com/InternLM/HuixiangDou)，从中将文件操作、向量提取及存储、问题查询等子模块抽离，基于本项目需求改造。主要相关文件如下：
 - [feature_store.py](feature_store.py)--基本复用茴香豆[feature_store.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/feature_store.py)文件
 - [retriever.py](retriever.py)--基本复用茴香豆[retriever.py](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/retriever.py)文件


&emsp;&emsp;茴香豆具有拒答功能，考虑到ComfyUI社区极其活跃，当前收集资料必然不全，故保留拒答功能，通过其判断当前数据未覆盖的问题，直接用LLM推理生成答案。基于茴香豆构建RAG的过程极大加深对RAG架构了解，是很好的学习经验，感兴趣的朋友可以从茴香豆中学习。

 &emsp;&emsp;目前搜集了6000多ComfyUI相关文档，基本均是markdown文档，其中英文和中文数量差不多，都是3000多。RAG构建中需要从知识库文档中抽取feature并保存，当前方案与茴香豆保持一致，使用Faiss实现。此过程相关文件如下所述：
 - [source/knowledges](source/knowledges)--原始文档存储路径，分zh和en，即对应中文文档和英文文档；中文文档目前是comfyui-nodes-docs中相关文档，而英文文档目前是ComfyUI-docs和SaltAI-Web-Docs中相关文档
 - [source/questions](source/questions)--存放用于计算RAG据答问题阈值的问题文件，也分zh和en
 - [source/workdir](source/workdir)--存放[feature_store.py](feature_store.py)处理结果，也是分zh和en
   - db_reject--使用Faiss抽取的RAG据答vectorstore
   - db_response--使用Faiss抽取的RAG回答vectorstore
   - preprocess--feature_store.py中会先将knowledges中的文档预处理，然后存放在此路径中

**注意**：当文档数量比较多时，Faiss会很慢，本项目在茴香豆原始的feature_store.py中进行了修改，先所有文件分成n份，然后使用多线程基于Faiss抽取特征并保存，具体部分可见代码[ingress_reject后部分](feature_store.py#411)。有两点需要注意，一是执行一次faiss抽取保存后，会在路径下存储两个文件，分别是index.faiss和index.pkl，index是默认值，可以使用index_name指定，如[zh/db_reject](source/workdir/zh/db_reject)中因为当时将所有文件分为4份，故此路径下有index1~4类文件，然后再使用merge_from方法将4个faiss文件合并成一个faiss文件，如下
- index.faiss/index.pkl--这两个文件是由faiss-cpu库基于merge_from方法将index1~4合并而来
- index-gpu.faiss/index-gpu.pkl--这两个文件是使用faiss-gpu库，修改langchain_community.vectorstores.faiss.Faiss.merge_from中少量代码将index1~4合并而来

 &emsp;&emsp;出现上述两组faiss文件的原因是faiss库gpu版本和cpu版本区别造成的，目前faiss-gpu库不支持merge_from方法，而faiss-cpu库支持，而后续在langchain的一个issue中发现了使用faiss-gpu库时的处理方法，故分别生成了两组文件，使用时直接通过index_name指定名称即可加载对应文件。参考文件如下:
 - [langchain_community.vectorstores.faiss.FAISS](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
 - [merge two FAISS indexes](https://github.com/langchain-ai/langchain/issues/1447)


## LLM推理
&emsp;&emsp;LLM推理部分也主要借鉴茴香豆，llm_infer.py中将调用远程LLM接口和本地LLM模型推理集成在一起，可通过config.ini文件进行配置
 - [llm_infer.py](llm_infer.py)--主要参考以下两个文件构建复合LLM推理类，可有效基于本地模型或调用远端接口进行推理
   - [llm_client.py](source/llm_client.py)--茴香豆中客户端调用的llm[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_client.py)，本文件进行了适当注释，可参考
   - [llm_server_hybrid.py](source/llm_server_hybrid.py)--茴香豆中llm的服务端[接口](https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/llm_server_hybrid.py)，本文件进行了适当注释，可参考

## ASR模块
&emsp;&emsp;本模块将语音转换为文本，使用基于[openai whisper](https://github.com/openai/whisper)衍生的两种方案，分别是[whispercpp](https://github.com/AIWintermuteAI/whispercpp)和[whisperX](https://github.com/m-bain/whisperX)，两种方案表现区别不大，whispercpp底层基于[whisper.cpp](https://github.com/ggerganov/whisper.cpp)，whisperX底层基于[faster-whisper](https://github.com/guillaumekln/faster-whisper)，均用C/C++加速计算；但从安装便捷性上，推荐使用whisperX

**跟新：**综合考虑效果、部署便捷醒等因素，调整[demo](demo.py)ASR模块设置，直接使用whisperX

## TTS模块
&emsp;&emsp;本模块将LLM推理生成文本转为语音，测试了[Chattts](https://github.com/2noise/ChatTTS)和[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，两者均是今年很火的TTS模型。Chattts效果测试下来，应该算是一般，并且生成的语音中会出现输入文本中不存在的词汇；GPT-SoVITS功能效果很好，且可以使用开源的社区角色模型生成不同的音色，推荐使用GPT-SoVITS生成语音。

**跟新：**综合考虑效果、部署便捷醒等因素，调整[demo](demo.py)TTS模块设置，直接使用GPT-SoVITS生成语音

## 生图模块
&emsp;&emsp;[ComfyUI](https://github.com/comfyanonymous/ComfyUI)提供了完整、灵活或者说单调、高效的api接口，在前端界面设置中开启开发者模型就能激活“保存API”功能，能将在界面上跑通的任何workflow保存为ComfyUI api能直接调用的workflow_api.json文件，而其内部就是各种节点实际参数排列组成，故可以人为构造符合ComfyUI api接口的workflow对象，与Gradio结合就能实现类似Stable Diffusion WebUI的界面。当前服务中通过上述方法实现了具有几个常规、较固定的生成工作流前端界面，感兴趣的朋友可以参考[module_comfyui.py](module_comfyui.py)。基于Gradio构建的固定workflow并不能展现ComfyUI的全部能力，故基于Gradio的页面加载能力，将ComfyUI的前端页面直接集成到了项目页面，使用者可在对话界面查询完问题后，直接在ComfyUI界面中进行验证。