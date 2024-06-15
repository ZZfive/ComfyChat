# 数据说明

- [v1](#v1)
- [v2](#v2)

# v1
&emsp;&emsp;v1数据集构建时，是通过[https://github.com/BlenderNeko/ComfyUI-docs](https://github.com/BlenderNeko/ComfyUI-docs)中维护的自定义列表从github拉去了各自定义节点的说明文档，并使用deepseek、kimi等LLMs构建了中英文微调数据集，和Aplaca-GPT4数据集混合构建了数据集v1。构建过程中有多个conversations形式的json文件，在此说明：
 - comfyui_node_data.json--针对三个意图问题分别构建了多个不同问题模板，然后对每个自定义节点随机抽取模板中一个组合一个对问数据，最终汇总构成
 - comfyui_node_data_zh.json--使用deepseek或kimi对comfyui_node_data.json翻译为中文
 - comfyui_node_data_together.json--comfyui_node_data.json中针对每个自定义接单的三个意图问题都是单独构建一个conversation，而本文件是将每个自定义节点中的三个问题放在了一个conversation中
 - comfyui_data_v1.json--最终用于训练v1模型的数据；由完整的comfyui_node_data.json、40%的alpacha_gpt4_data_modified.json、custom_nodes_mds路径下所有自定义节点中构建的final.json中的数据以及少数关于comfyui的说明数据构成conversation形式的数据
 - alpacha_gpt4_data.json--原始数据
 - alpacha_gpt4_data_modified.json--调整后的conversation形式的文件
 - alpacha_gpt4_data_zh.json--使用deepseek或kimi对alpacha_gpt4_data.json翻译为中文
 - alpacha_gpt4_data_zh_modified.json--调整后的conversation形式的文件

# v2