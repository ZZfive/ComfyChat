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
import time
from typing import Any

import pypandoc
from openai import OpenAI

from utils import get_data_from_url, save2json, load4json
import config


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
                                         together: bool = False, save_path: str = '/root/code/ComfyChat/data/comfyui_node_data.json') -> Any: 
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
        save2json(data, save_path)
    except Exception as e:
        raise ValueError(f"err: {e}")
    

def eng2zh_moonshot(eng_text):
    client = OpenAI(api_key=config.MOONSHOT_API_KEY, base_url="https://api.moonshot.cn/v1")

    completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system", "content": "You are a master of Chinese-English translation. Please accurately translate the subsequent English text into Chinese. Some proper nouns can be retained without translation."},
        {"role": "user", "content": f"Translate the following text into Chinese: {eng_text}"}
    ],
    temperature=0.3,
    )
    ans = completion.choices[0].message.content
    print('英文：', eng_text)
    print('中文：', ans)
    print('*' * 40)
    time.sleep(20)
    return ans


def eng2zh_deepseek(eng_text: str) -> str:
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a master of Chinese-English translation. Please accurately translate the subsequent English text into Chinese. Some proper nouns can be retained without translation."},
        {"role": "user", "content": f"Translate the following text into Chinese: {eng_text}"}
    ],
    )
    ans = completion.choices[0].message.content
    print('英文：', eng_text)
    print('中文：', ans)
    print('*' * 40)
    time.sleep(20)
    return ans
    

def construct_data_zh_from_custom_node_list(custom_node_list_url: str = 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json',
                                         custom_node_map_url: str = 'https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json',
                                         together: bool = False, save_path: str = '/root/code/ComfyChat/data/comfyui_node_data_zh.json') -> Any: 
    random.seed(42)
    try:
        data = []
        custom_node_list = get_data_from_url(custom_node_list_url)['custom_nodes']
        custom_node_map = get_data_from_url(custom_node_map_url)

        describe_tempaltes = ["### 的用途是什么？", "### 能做什么？", "### 的主要功能是什么？",
                              "### 有哪些特点？", "### 与其他自定义节点有什么不同？",
                              "### 的能力是什么？", "### 的主要应用是什么？",
                              "您能解释一下 ### 的关键特性吗？", "什么特性使得 ### 与其他自定义节点区别开来？",
                              "### 可以帮助解决哪些问题或挑战？", "在什么场景或背景下可以使用 ### ？"]
        node_tempaltes = ["### 包含哪些节点？", "### 里有哪些节点？", "您能提供一份在 ### 内可用的节点列表吗？"
                          "### 包含了哪些组件或模块？", "通过 ### 可以访问哪些节点？",
                          "构成 ### 的各个节点是什么？", "您能概述一下 ### 内的节点结构或组织吗？",
                          "### 包含多少个节点，它们分别是什么？", "### 的核心节点是什么？"]
        link_tempaltes = ["### 的 GitHub 地址是什么？", "您能提供 ### 的 GitHub 仓库链接吗？",
                          "我在 GitHub 的哪里可以找到 ### 的源代码？", "托管 ### 的 GitHub 仓库的 URL 是什么？",
                          "您能分享一下 ### 项目的 GitHub 地址吗？", "我如何找到 ### 的 GitHub 页面？"]

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
            assistant1["content"] = f"{node_name} 的作者是 {node_author}，{eng2zh_deepseek(node_description) if node_description else f'为收集到其他具体信息'}"

            user2 = {}
            user2["role"] = "user"
            user2["content"] = random.choice(node_tempaltes).replace("###", node_name)
            assistant2 = {}
            assistant2["role"] = "assistant"
            assistant2["content"] = f"{node_name} 包含{len(node_map)}个节点，分别是 {node_map}" if node_map else f"{node_name} 没有具体节点"

            user3 = {}
            user3["role"] = "user"
            user3["content"] = random.choice(link_tempaltes).replace("###", node_name)
            assistant3 = {}
            assistant3["role"] = "assistant"
            assistant3["content"] = f"{node_name} 的Github地址是 {node_link}" if node_link else f"{node_name} 的Github地址未收集到"

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
        save2json(data, save_path)
    except Exception as e:
        raise ValueError(f"err: {e}")


def alpaca_modify(alpaca_path: str, save_path: str) -> None:
    try:
        data = []
        alpaca_data = load4json(alpaca_path)
        for val in alpaca_data:
            user = {}
            user["role"] = "user"
            user["content"] = val["instruction"] + val["input"]
            assistant = {}
            assistant["role"] = "assistant"
            assistant["content"] = val["output"]
            messages = []
            messages.append(user)
            messages.append(assistant)
            data.append({"messages": messages})
        save2json(data, save_path)
    except Exception as e:
        raise ValueError(f"err: {e}")


def get_data_from_openrouter(md_path: str, model: str = "google/gemma-7b-it:free") -> str:
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
    )

    template = '''
    I need to build a llm fine-tuning dataset. You need to understand the content of the document I input, then construct several pairs of question and answer data yourself, and return them in json format.\n---\nOnly question and answer data in json format is returned. The returned content is as follows: {0}
    '''

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print(template.format(md_content))

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": template.format(md_content)},
    ],
    temperature=1.0,
    )
    print(completion.choices[0].message.content)


if __name__=='__main__':
    # md2txt()
    # construct_data_from_custom_node_list(together=True, seve_path='/root/code/ComfyChat/data/comfyui_node_data_together.json')
    # alpaca_modify('/root/code/ComfyChat/data/alpaca_gpt4_data_zh.json', '/root/code/ComfyChat/data/alpaca_gpt4_data_zh_modification.json')
    
    # construct_data_zh_from_custom_node_list()

    # eng_text = "The author of FreeU_Advanced is WASasquatch, This custom node provides advanced settings for FreeU."
    # eng2zh_deepseek(eng_text)

    get_data_from_openrouter('/root/code/ComfyChat/data/custom_nodes_mds/a-person-mask-generator/README.md')