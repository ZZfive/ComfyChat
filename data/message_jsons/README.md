# 数据说明

- [v1](#v1)
- [v2](#v2)

# v1
&emsp;&emsp;v1数据集构建时，是通过[ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)中维护的自定义列表从github拉去了各自定义节点的项目文档，并使用deepseek、kimi等LLMs接口构建了中英文微调数据集，和Aplaca-GPT4数据集混合构建了数据集v1。构建过程中有多个conversations形式的json文件，在此说明：
 - comfyui_node_data.json--针对三个意图问题分别构建了多个不同问题模板，然后对每个自定义节点随机抽取模板中一个组合一个对问数据，最终汇总构成
 - comfyui_node_data_zh.json--使用deepseek或kimi将comfyui_node_data.json翻译为中文后的文件
 - comfyui_node_data_together.json--comfyui_node_data.json中针对每个自定义节点的三个意图问题都是单独构建一个conversation，而本文件是将每个自定义节点中的三个问题放在了一个conversation中
 - comfyui_data_v1.json--最终用于训练v1模型的数据；由完整的comfyui_node_data.json、40%的alpacha_gpt4_data_modified.json、custom_nodes_mds路径下所有自定义节点中构建的final.json中的数据以及少数关于comfyui的说明数据构成conversation形式的数据混合构成
 - alpacha_gpt4_data.json--原始文件
 - alpacha_gpt4_data_modified.json--调整为conversation形式的文件
 - alpacha_gpt4_data_zh.json--使用deepseek或kimi对alpacha_gpt4_data.json翻译为中文的文件
 - alpacha_gpt4_data_zh_modified.json--调整为conversation形式的文件

# v2
&emsp;&emsp;除了v1中单独从每个自定义节点github仓库中拉取的md文档，还有一些社区开发者集中为大量节点提供文档的项目，可以从中提取出大量的针对各类自定义节点的数据。文件说明如下：
 - comflowy_en.json--从comflowy中构建的英文问答数据
 - comflowy_zh.json--从comflowy中构建的中文问答数据
 - ComfyUI-docs.json--从ComfyUI-docs中构建的数据，英文
 - comfyui-nodes-docs.json--从comfyui-nodes-docs中构建的数据，中文
 - SaltAI-Web-Docs.json--congSaltAI-Web-Docs中构建的数据，英文
 - community_en.json--英文conversations（22233）；将comflowy_en.json，ComfyUI-docs.json和SaltAI-Web-Docs.json三个文件组合
 - community_zh.json--中文conversations（14914）；comflowy_zh.json和comfyui-nodes-docs.json两个文件组合
 - comfyui_data_v2_1.json--v2场景下的第一版数据集，直接将comfyui_data_v1.json和community_en.json混合，有48960条messages