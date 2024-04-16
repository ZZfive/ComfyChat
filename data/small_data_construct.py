'''
基于当前收集的md信息构建第四课微调的数据集
    1、数据清洗
        内容信息少的markdown直接删除
        markdown中的表、code片段等删除？
        markdown中的本体图片链接信息删除？
        其他清洗细节？
    2、基于RAG方案构建问答数据
        清洗后数据向量化
        构建问题列表
        部署茴香豆服务
        调用api生成回答
    3、二次清洗

另一种方案：现在有deepseek、kimi和openrouter上免费额度的api接口；mds路径下基本每个子目录就是一个节点
可以基于目录中markdown文件的质量先对每个节点构建三个问题
    节点用处
    节点安装方法
    节点其他特性？？

方法是在对markdown文件清洗后，可以将剩下的markdown构建提示词，输入上述api接口中，要求对应的LLM生成答案
    对于每个自定义节点下的文件，每个文件基于构建固定的prompt template要llm自行分析内容，要其分析内容，构建问答对
        如何安装
        有什么作用
        ？？？
'''

import os
import re
import random
from typing import Any

import pypandoc

from utils import get_data_from_url, save2json


def md2txt(mds_dir: str="/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    for item in os.listdir(mds_dir):
        item_path = os.path.join(mds_dir, item)
        if os.path.isdir(item_path):
            for md in os.listdir(item_path):
                if md.endswith('.MD') or md.endswith('.MDX') or md.endswith('.md') or md.endswith('.mdx'):
                    md_path = os.path.join(item_path, md)
                    txt_path = re.sub(r'\.(md|mdx|MD|MDX)$', '.txt', md_path)
                    try:
                        # 将Markdown文件转换为纯文本
                        output = pypandoc.convert_file(md_path, 'plain', format='markdown')  # 需要安装pandoc，此转换会丢失md中的外链信息
                        # 将转换后的纯文本保存到文件
                        with open(txt_path, 'w') as f:
                            f.write(output)
                    except Exception as e:
                        print(f"Conversion of MD file {md_path} to txt failed, error {e}")


def construct_data_from_custom_node_list(custom_node_list_url: str = 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json',
                                         custom_node_map_url: str = 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json',
                                         together: bool = False, seve_path: str = '/root/code/ComfyChat/data/comfyui_node_data.json') -> Any: 
    random.seed(42)
    try:
        data = []
        custom_node_list = get_data_from_url(custom_node_list_url)['custom_nodes']
        custom_node_map = get_data_from_url(custom_node_map_url)

        describe_tempaltes = ["What is the use of ###?", "What ### can do?", "What is the main function of ###?",
                              "What are the features of ###?", "What makes ### different from other custom nodes?",
                              "What are the capabilities of ###?", "What are the primary applications of ###?",
                              "Can you explain the key features of ###?", "What distinguishes ### from other custom nodes?",
                              "What problems or challenges can ### help solve?", "In what scenarios or backgrounds can ### be used?"]
        node_tempaltes = ["What nodes does ### contain?", "What nodes are in ###?", "Could you provide a list of nodes available within ###?"
                          "What are the components or modules included in ###?", "Which nodes can be accessed through ###?",
                          "What are the individual nodes that make up ###?", "Can you outline the node structure or organization within ###?",
                          "How many nodes are contained in ###, and what are they?", "What are the core nodes of ###?"]
        link_tempaltes = ["What is the github address of the ###?", "Could you provide the GitHub repository link for ###?",
                          "Where can I find the source code for ### on GitHub?", "What is the URL of the GitHub repository where ### is hosted?",
                          "Can you share the GitHub address where I can access the ### project?", "How do I locate the GitHub page for ###?"]

        for node in custom_node_list:
            node_name = node["title"]
            node_author = node["author"]
            node_description = node["description"]
            node_map = []
            for url in node["files"]:
                if url in custom_node_map:
                    node_map += custom_node_map[url][0]
            node_link = node["reference"]

            user1 = {}
            user1["role"] = "user"
            user1["content"] = random.choice(describe_tempaltes).replace("###", node_name)
            assistant1 = {}
            assistant1["role"] = "assistant"
            assistant1["content"] = f"The author of {node_name} is {node_author}, {node_description if node_description else 'No other specific information was collected'}"

            user2 = {}
            user2["role"] = "user"
            user2["content"] = random.choice(node_tempaltes).replace("###", node_name)
            assistant2 = {}
            assistant2["role"] = "assistant"
            assistant2["content"] = f"{node_name} contains {len(node_map)} nodes, where there are {node_map}" if node_map else f"{node_name} has no specific node"

            user3 = {}
            user3["role"] = "user"
            user3["content"] = random.choice(link_tempaltes).replace("###", node_name)
            assistant3 = {}
            assistant3["role"] = "assistant"
            assistant3["content"] = f"The github address of the {node_name} is {node_link}" if node_link else f"The github link of {node_name} was not collected"

            if together:
                messages = []
                messages.append(user1)
                messages.append(assistant1)
                messages.append(user2)
                messages.append(assistant2)
                messages.append(user3)
                messages.append(assistant3)
                data.append({"messages": messages})
            else:
                messages = []
                messages.append(user1)
                messages.append(assistant1)
                data.append({"messages": messages})

                messages = []
                messages.append(user2)
                messages.append(assistant2)
                data.append({"messages": messages})

                messages = []
                messages.append(user3)
                messages.append(assistant3)
                data.append({"messages": messages})
        save2json(data, seve_path)
    except Exception as e:
        raise ValueError(f"err: {e}")


if __name__=='__main__':
    # md2txt()
    construct_data_from_custom_node_list(together=True, seve_path='/root/code/ComfyChat/data/comfyui_node_data_together.json')