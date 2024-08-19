import os
import re
import random
import time
from typing import Any, List, Dict

import requests
import pypandoc
from openai import OpenAI

from utils import get_data_from_url, save2json, load4json, parse_json, create_logger, extract_name_extension
from prompt_templates import *
import config


# 将拉取的自定义节点中的markdown文档转为txt文件并保存
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


# 针对三个问题构建了一些问题模板，然后和comfyui-manager中维护的自定义节点排列组合构建一个对话数据json文件--感觉意义不大
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

    # completion = client.chat.completions.create(
    # model="moonshot-v1-8k",
    # messages=[
    #     {"role": "system", "content": "You are a master of Chinese-English translation. Please accurately translate the subsequent English text into Chinese. Some proper nouns can be retained without translation."},
    #     {"role": "user", "content": f"Translate the following text into Chinese: {eng_text}"}
    # ]
    # )

    completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
        {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
    ],
    )
    ans = completion.choices[0].message.content
    # print('英文：', eng_text)
    # print('中文：', ans)
    # print('*' * 40)
    time.sleep(30)
    return ans


def eng2zh_deepseek(eng_text: str) -> str:
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    # completion = client.chat.completions.create(
    # model="deepseek-chat",
    # messages=[
    #     {"role": "system", "content": "You are a master of Chinese-English translation. Please accurately translate the subsequent English text into Chinese. Some proper nouns can be retained without translation."},
    #     {"role": "user", "content": f"Translate the following text into Chinese, don't add anything irrelevant: {eng_text}"}
    # ],
    # )

    completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
        {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
    ],
    )
    ans = completion.choices[0].message.content
    # print('英文：', eng_text)
    # print('中文：', ans)
    # print('*' * 40)
    time.sleep(20)
    return ans


def eng2zh_openrouter(eng_text: str) -> str:
    client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    # completion = client.chat.completions.create(
    # model="google/gemma-7b-it:free",
    # messages=[
    #     {"role": "system", "content": "You are a master of Chinese-English translation. Please accurately translate the subsequent English text into Chinese. Some proper nouns can be retained without translation."},
    #     {"role": "user", "content": f"Translate the following text into Chinese, don't add anything irrelevant: {eng_text}"}
    # ],
    # )

    completion = client.chat.completions.create(
    model="google/gemma-7b-it:free",
    messages=[
        {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
        {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
    ],
    )
    ans = completion.choices[0].message.content
    # print('英文：', eng_text)
    # print('中文：', ans)
    # print('*' * 40)
    # time.sleep(20)
    return ans


def eng2zh_siliconflow(eng_text: str, model: str = 'deepseek-ai/deepseek-v2-chat') -> str:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
            {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
        ]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {config.SILICONFLOW_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers).json()

    ans = response['choices'][0]['message']['content']
    
    return ans
    

# 与construct_data_from_custom_node_list的区别就是使用的是中文模板
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


# 将alpacha数据集转为openai对话类型结构数据
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


def get_data_from_openrouter(subject: str, md_path: str, model: str = "google/gemma-7b-it:free",
                             system_prompt: str = system_prompt1, template: str = template1) -> str:
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
    )

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # print(template.format(md_content))

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.format(subject, md_content)},
    ],
    temperature=1.0,
    )
    return completion.choices[0].message.content


def get_data_from_deepseek(subject: str, md_path: str, model: str = "deepseek-chat",
                           system_prompt: str = system_prompt1, template: str = template1) -> str:
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # print(template.format(md_content))

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.format(subject, md_content)},
    ],
    temperature=1.0,
    )
    return completion.choices[0].message.content


def get_data_from_moonshot(subject: str, md_path: str, model: str = "moonshot-v1-8k",
                           system_prompt: str = system_prompt1, template: str = template1) -> str:
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(api_key=config.MOONSHOT_API_KEY, base_url="https://api.moonshot.cn/v1")

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # print(template.format(md_content))

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.format(subject, md_content)},
    ],
    temperature=1.0,
    )
    return completion.choices[0].message.content


def constrcut_data_from_md(md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds",
                           successful_node_list_path: str = "/root/code/ComfyChat/data/successful_node_list.json",
                           unsuccessful_node_list_path: str = "/root/code/ComfyChat/data/unsuccessful_node_list.json") -> None:
    logger = create_logger('constrcut_data_from_md')

    if os.path.exists(successful_node_list_path):
        successful_nodes = load4json(successful_node_list_path, [])
    else:
        with open(successful_node_list_path, "a"):
            os.utime(successful_node_list_path, None)
        successful_nodes = []

    if os.path.exists(unsuccessful_node_list_path):
        unsuccessful_nodes = load4json(unsuccessful_node_list_path, [])
    else:
        with open(unsuccessful_node_list_path, "a"):
            os.utime(unsuccessful_node_list_path, None)
        unsuccessful_nodes = []

    updated_successful_nodes = []
    updated_unsuccessful_nodes = []
    try:
        for item in os.listdir(md_base_dir):
            node_dir = os.path.join(md_base_dir, item)
            if os.path.isdir(node_dir) and len(os.listdir(node_dir)) > 0:
                try:
                    for md in os.listdir(node_dir):
                        if md.endswith('.MD') or md.endswith('.MDX') or md.endswith('.md') or md.endswith('.mdx'):
                            md_path = os.path.join(node_dir, md)
                            if md_path not in successful_nodes:
                                md_name = os.path.splitext(md)[0]
                                rsp = ''
                                rsp = get_data_from_deepseek(item, md_path)
                                # time.sleep(10)
                                rsp_json = parse_json(rsp)
                                save2json(rsp_json, os.path.join(node_dir, f"{md_name}.json"))
                                updated_successful_nodes.append(md_path)
                                logger.info(f'Successfully extracting data from file: {md_path}')
                except Exception as e:
                    logger.error(f"Constrcut data of MD file: ###{md_path}### to txt failed, error {e}")
                    updated_unsuccessful_nodes.append({md_path: rsp})
    finally:
        successful_nodes += updated_successful_nodes
        unsuccessful_nodes += updated_unsuccessful_nodes
        save2json(successful_nodes, successful_node_list_path)
        save2json(unsuccessful_nodes, unsuccessful_node_list_path)


def construct_single_messages(user_content: str, assistant_content: str) -> Dict[str, List[Dict[str, str]]]:
    user = {}
    user["role"] = "user"
    user["content"] = user_content
    assistant = {}
    assistant["role"] = "assistant"
    assistant["content"] = assistant_content
    messages = []
    messages.append(user)
    messages.append(assistant)
    return {"messages": messages}


def parse_data_from_md_json(md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    logger = create_logger('parse_data_from_md_json')
    for item in os.listdir(md_base_dir):
        node_dir = os.path.join(md_base_dir, item)
        print('*' * 80)
        print(node_dir)
        if os.path.isdir(node_dir) and len(os.listdir(node_dir)) > 0 and 'final.json' not in os.listdir(node_dir):
            data = []
            try:
                for md in os.listdir(node_dir):
                    try: 
                        if md.endswith('.json'):
                            path = os.path.join(node_dir, md)
                            qa = load4json(path)
                            if isinstance(qa, list) and isinstance(qa[0], dict):
                                list_dict = qa
                                for v in list_dict:
                                    user_content = v.get('input', None) or v.get('question', None) or v.get('Question', None) or v.get('question_text', None) or v.get('prompt', None)
                                    assistant_content = v.get('output', None) or v.get('answer', None) or v.get('Answer', None) or v.get('answer_text', None) or v.get('completion', None)
                                    if user_content is not None and assistant_content is not None:
                                        data.append(construct_single_messages(user_content, assistant_content))
                                    else:
                                        logger.info(f"{v}")
                            elif isinstance(qa, dict):
                                keys = list(qa.keys())
                                if isinstance(qa[keys[0]], list) and isinstance(qa[keys[0]][0], dict):
                                    list_dict = qa[keys[0]]
                                    for v in list_dict:
                                        user_content = v.get('input', None) or v.get('question', None) or v.get('Question', None) or v.get('question_text', None) or v.get('prompt', None)
                                        assistant_content = v.get('output', None) or v.get('answer', None) or v.get('Answer', None) or v.get('answer_text', None) or v.get('completion', None)
                                        if user_content is not None and assistant_content is not None:
                                            data.append(construct_single_messages(user_content, assistant_content))
                                        else:
                                            logger.info(f"{v}")
                                elif isinstance(qa[keys[0]], str):
                                    if len(keys) % 2 == 0:
                                        for i in range(0, len(keys), 2):
                                            data.append(construct_single_messages(qa[keys[i]], qa[keys[i+1]]))
                                    else:
                                        raise ValueError(f'问答数据个数为基数')
                    except Exception as e:
                        logger.error(f"Error happened: {path}, error: {e}")
                        pass
            finally:
                if data:
                    save2json(data, os.path.join(node_dir, 'final.json'))
                    logger.info(f"{os.path.join(node_dir, 'final.json')} saved \n")


def semi_automatic_for_one_node1(node_name: str, questions: List[str], answers: List[str],
                                md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    assert len(questions) == len(answers)
    save_path = os.path.join(os.path.join(md_base_dir, node_name), 'final.json')
    if os.path.exists(save_path):
        data = load4json(save_path, [])
    else:
        data = []
    for i in range(len(questions)):
        data.append(construct_single_messages(questions[i], answers[i]))
    save2json(data, save_path)


def semi_automatic_for_one_node2(node_name: str, qa,
                                md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    
    save_path = os.path.join(os.path.join(md_base_dir, node_name), 'final.json')
    if os.path.exists(save_path):
        data = load4json(save_path, [])
    else:
        data = []
    
    if isinstance(qa, dict):
        keys = list(qa.keys())
        if isinstance(qa[keys[0]], list) and isinstance(qa[keys[0]][0], dict):
            list_dict = qa[keys[0]]
            for v in list_dict:
                user_content = v.get('input', None) or v.get('question', None) or v.get('Question', None) or v.get('question_text', None) or v.get('prompt', None)
                assistant_content = v.get('output', None) or v.get('answer', None) or v.get('Answer', None) or v.get('answer_text', None) or v.get('completion', None)
                if user_content is not None and assistant_content is not None:
                    data.append(construct_single_messages(user_content, assistant_content))
    save2json(data, save_path)


# 校验最终的messages数据集结构是否正常
def check_messages_json(md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    for item in os.listdir(md_base_dir):
        node_dir = os.path.join(md_base_dir, item)
        if os.path.isdir(node_dir) and len(os.listdir(node_dir)) > 0 and 'final.json' in os.listdir(node_dir):
            final_path = os.path.join(node_dir, "final.json")
            final_data = load4json(final_path)
            if isinstance(final_data, list):
                for v in final_data:
                    if isinstance(v, dict) and "messages" in v and len(v["messages"]) == 2:
                        d1 = v["messages"][0]
                        d2 = v["messages"][1]
                        if ("role" in d1 and d1["role"] and "content" in d1 and d1["content"] and isinstance(d1["content"], str)) and ("role" in d2 and d2["role"] and "content" in d2 and d2["content"] and isinstance(d2["content"], str)):
                            continue
                        else:
                            print(v)
                            print('错误：', final_path)
                    else:
                        print(v)
                        print('错误：', final_path)
            else:
                print('错误：', final_path)


def extract_text_before_newline(text: str) -> str:
    index = text.find('\n\n')
    if index == -1:
        index = text.find('\n')
    if index != -1:
        return text[:index]
    return text


# 将custom_nodes_mds路径下每个节点中的final_zh.json中的内容翻译为英文
def translate_final2zh(md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    logger = create_logger('translate_final')
    num = 0
    for item in os.listdir(md_base_dir):
        node_dir = os.path.join(md_base_dir, item)
        if not os.path.isdir(node_dir) or len(os.listdir(node_dir)) == 0 or 'final.json' not in os.listdir(node_dir) or 'final_zh.json' in os.listdir(node_dir):
            continue
        # if os.path.isdir(node_dir) and len(os.listdir(node_dir)) > 0 and 'final.json' in os.listdir(node_dir) and 'final_zh.json' not in os.listdir(node_dir):
        final_path = os.path.join(node_dir, "final.json")
        final_data = load4json(final_path)
        final_zh = []
        for v in final_data:
            try:
                user_content = v["messages"][0]["content"]
                # user_content_zh = extract_text_before_newline(eng2zh_deepseek(user_content))
                user_content_zh = extract_text_before_newline(eng2zh_moonshot(user_content))
                # user_content_zh = eng2zh_openrouter(user_content)
                assistant_content = v["messages"][1]["content"]
                # assistant_content_zh = extract_text_before_newline(eng2zh_deepseek(assistant_content))
                assistant_content_zh = extract_text_before_newline(eng2zh_moonshot(assistant_content))
                # assistant_content_zh = eng2zh_openrouter(assistant_content)
                final_zh.append(construct_single_messages(user_content_zh, assistant_content_zh))
            except Exception as e:
                logger.error(f"file: {final_data}, message: {v}, error: {e}")
        if final_zh:
            save2json(final_zh, os.path.join(node_dir, "final_zh.json"))
            logger.info(f"{os.path.join(node_dir, 'final_zh.json')} translated successfully \n")
            num += 1
        # break
    logger.info(f'num of final_zh.json is {num}')


# 基于alpaca、由custom_node_list人工构建和使用deepseek从收集的mds中生成的数据构建一个训练数据集
def construct_data(save_path: str, ratio: float = 0.4,
                   md_base_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
    random.seed(42)

    comfychat_data = {
        "questions": [
            "Could you introduce yourself?",
            "What is your identity?",
            "Can you tell me more about who you are?",
            "Please explain your role.",
            "Who am I chatting with?",
            "What's your purpose?",
            "How would you describe yourself?",
            "Tell me more about your capabilities."
        ],
        "answers": [
            "I am ComfyChat, an LLM-based smart assistant ready to help you with your ComfyUI queries.",
            "My identity is ComfyChat, an intelligent assistant designed to answer your questions about ComfyUI.",
            "As an LLM-based assistant, I am here to help you find answers to any questions you might have about ComfyUI.",
            "My role is to provide you with assistance and information related to ComfyUI as an LLM-based intelligent assistant.",
            "You're chatting with ComfyChat, a helpful AI designed to support you with ComfyUI-related inquiries.",
            "My purpose is to offer guidance, information, and assistance on ComfyUI topics.",
            "I'm an AI assistant based on LLM, dedicated to answering your questions and providing help with ComfyUI.",
            "My capabilities include understanding your queries and providing accurate information about ComfyUI to the best of my knowledge."
        ]
    }

    comfyui_data = {
        "questions": [
            "What is ComfyUI, and what are its primary purposes?",
            "What makes ComfyUI unique compared to other Stable Diffusion web UIs?",
            "How can I install and set up ComfyUI for optimal performance?",
            "What are the key features and functionalities offered by ComfyUI?",
            "How can I leverage ComfyUI for various image generation and editing tasks?",
            "What customization options does ComfyUI provide to enhance user experience?",
            "How does ComfyUI ensure compatibility with different Stable Diffusion models and extensions?",
            "What are the system requirements and dependencies for running ComfyUI?"
        ],
        "answers": [
            "ComfyUI is an advanced web user interface for Stable Diffusion, designed to offer a customizable, user-friendly experience with a wide range of features for text-to-image and image-to-image generation, as well as in-painting and out-painting tasks.",
            "ComfyUI differentiates itself by providing a highly customizable layout, support for advanced prompt editing, and numerous options for fine-tuning image generation processes, making it an adaptable solution for both beginners and experienced users.",
            "To install ComfyUI, clone the GitHub repository, create a dedicated Python environment, install required dependencies, and configure settings to optimize performance based on your system specifications and preferences.",
            "Key features and functionalities of ComfyUI include dynamic prompting, advanced prompt editing, support for various image-to-image and in-painting techniques, scriptable workflows, prompt templates, and compatibility with custom Stable Diffusion models.",
            "Leverage ComfyUI's capabilities by utilizing its numerous tools and options to generate images from text prompts, apply desired styles to images, refine results through in-painting or out-painting, and automate tasks using custom scripts.",
            "ComfyUI offers a wide array of customization options, including the ability to adjust widget positioning, apply different color themes, create and save custom workflows, and integrate external tools or extensions to enhance functionality.",
            "ComfyUI ensures compatibility with various Stable Diffusion models and extensions by staying up-to-date with the latest advancements, providing continuous support, and offering clear documentation for integration and troubleshooting.",
            "ComfyUI's system requirements include a modern web browser, Python 3.10 or higher, and the necessary dependencies, such as Git and pip. The recommended hardware depends on the complexity of your workflows, with higher VRAM GPUs and more RAM enabling smoother performance."
        ]
    }
    
    data = []

    for q, a in zip(comfychat_data['questions'], comfychat_data['answers']):
        data.append(construct_single_messages(q, a))

    for q, a in zip(comfyui_data['questions'], comfyui_data['answers']):
        data.append(construct_single_messages(q, a))

    data = data * 3
    # print(len(data))

    comfyui_node_data = load4json("/root/code/ComfyChat/data/comfyui_node_data.json")
    data += comfyui_node_data
    # print(len(data))

    alpach_data = load4json("/root/code/ComfyChat/data/alpaca_gpt4_data_modification.json")
    alpach_data = random.sample(alpach_data, int(ratio * len(alpach_data)))
    data += alpach_data
    # print(len(data))

    num = 0
    for item in os.listdir(md_base_dir):
        node_dir = os.path.join(md_base_dir, item)
        if os.path.isdir(node_dir) and len(os.listdir(node_dir)) > 0 and 'final.json' in os.listdir(node_dir):
            final_path = os.path.join(node_dir, "final.json")
            final_data = load4json(final_path)
            data += final_data
            num += 1
    # print(num)
    # print(len(data))

    random.shuffle(data)
    save2json(data, save_path)


# 基于项目chat2api[https://github.com/lanqian528/chat2api]调用openai免费gpt-3.5-turbo翻译
def eng2zh_chat2api(eng_text: str, model: str = 'gpt-3.5-turbo') -> str:
    url = "http://127.0.0.1:5005/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
            {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
        ]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {config.OPENAI_ACCESS_TOKEN}"
    }

    response = requests.post(url, json=payload, headers=headers).json()

    ans = response['choices'][0]['message']['content']
    
    return ans


# 基于项目chat2api[https://github.com/lanqian528/chat2api]调用openai免费gpt-3.5-turbo构建问答数据对
def get_data_from_chat2api(subject: str, md_path: str, model: str = 'gpt-3.5-turbo',
                           system_prompt: str = system_prompt1, template: str = template1) -> str:
    url = "http://127.0.0.1:5005/v1/chat/completions"

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.format(subject, md_content)}
        ]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {config.OPENAI_ACCESS_TOKEN}"
    }

    response = requests.post(url, json=payload, headers=headers)

    # if response.status_code == 200:
    ans = response.json()
    ans = ans['choices'][0]['message']['content']
    return ans


def get_data_from_siliconflow(subject: str, md_path: str, model: str = 'deepseek-ai/deepseek-v2-chat',
                              system_prompt: str = system_prompt1, template: str = template1) -> str:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.format(subject, md_content)}
        ]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {config.SILICONFLOW_API_KEY}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        # if response.status_code == 200:
        ans = response.json()
        # print(ans)
        ans = ans['choices'][0]['message']['content']
        return ans
    except Exception as e:
        raise ValueError(f"Error: {e}")


def generate_data_from_comfyui_docs(comfyui_docs_path: str = r"D:\git_github\self\ComfyChat\data\community_docs\repos\ComfyUI-docs\docs\Core Nodes",
                                    save_dir: str = r"D:\git_github\self\ComfyChat\data\community_docs\messages\ComfyUI-docs",
                                    successful_node_list_name: str = "successful_node_list.json",
                                    unsuccessful_node_list_name: str = "unsuccessful_node_list.json") -> None:
    logger = create_logger("generate_messages_from_comfyui_docs")

    successful_node_list_path = os.path.join(save_dir, "information", successful_node_list_name)
    if os.path.exists(successful_node_list_path):
        successful_nodes = load4json(successful_node_list_path, [])
    else:
        with open(successful_node_list_path, "a"):
            os.utime(successful_node_list_path, None)
        successful_nodes = []

    unsuccessful_node_list_path = os.path.join(save_dir, "information", unsuccessful_node_list_name)
    if os.path.exists(unsuccessful_node_list_path):
        unsuccessful_nodes = load4json(unsuccessful_node_list_path, [])
    else:
        with open(unsuccessful_node_list_path, "a"):
            os.utime(unsuccessful_node_list_path, None)
        unsuccessful_nodes = []

    try:
        for item in os.listdir(comfyui_docs_path):
            temp_path = os.path.join(comfyui_docs_path, item)
            if item != "media" and os.path.isdir(temp_path) and len(os.listdir(temp_path)) > 0:
                try:
                    for item2 in os.listdir(temp_path):
                        if item2 == "index.md" or item2 == "media":
                            continue
                        elif item2.endswith('.MD') or item2.endswith('.MDX') or item2.endswith('.md') or item2.endswith('.mdx'):
                            md_path = os.path.join(temp_path, item2)
                            md_name = os.path.splitext(item2)[0]
                            rsp = ''
                            rsp = get_data_from_chat2api(md_name, md_path)
                            time.sleep(2)
                            rsp_json = parse_json(rsp)
                            save2json(rsp_json, os.path.join(save_dir, f"{md_name}.json"))
                            successful_nodes.append(md_path)
                            logger.info(f'Successfully extracting data from file: {md_path}')
                            # break
                        elif os.path.join(temp_path, item2) and len(os.path.join(temp_path, item2)) > 0:
                            temp_path2 = os.path.join(temp_path, item2)
                            for item3 in os.listdir(temp_path2):
                                if item3 == "index.md" or item3 == "media":
                                    continue
                                elif item3.endswith('.MD') or item3.endswith('.MDX') or item3.endswith('.md') or item3.endswith('.mdx'):
                                    md_path = os.path.join(temp_path2, item3)
                                    md_name = os.path.splitext(item3)[0]
                                    rsp = ''
                                    rsp = get_data_from_chat2api(md_name, md_path)
                                    time.sleep(2)
                                    rsp_json = parse_json(rsp)
                                    save2json(rsp_json, os.path.join(save_dir, f"{md_name}.json"))
                                    successful_nodes.append(md_path)
                                    logger.info(f'Successfully extracting data from file: {md_path}')
                except Exception as e:
                    logger.error(f"{md_path}, error {e}")
                    unsuccessful_nodes.append(md_path)
    finally:
        save2json(successful_nodes, successful_node_list_path)
        save2json(unsuccessful_nodes, unsuccessful_node_list_path)


def generate_data_from_SaltAI_Web_Docs(SaltAI_Web_Docs_path: str = r"D:\git_github\self\ComfyChat\data\community_docs\repos\SaltAI-Web-Docs\docs\md",
                                       save_dir: str = r"D:\git_github\self\ComfyChat\data\community_docs\messages\SaltAI-Web-Docs",
                                       successful_node_list_name: str = "successful_node_list.json",
                                       unsuccessful_node_list_name: str = "unsuccessful_node_list.json") -> None:
    logger = create_logger("generate_data_from_comfyui_docs")

    successful_node_list_path = os.path.join(save_dir, "information", successful_node_list_name)
    if os.path.exists(successful_node_list_path):
        successful_nodes = load4json(successful_node_list_path, [])
    else:
        with open(successful_node_list_path, "a"):
            os.utime(successful_node_list_path, None)
        successful_nodes = []

    unsuccessful_node_list_path = os.path.join(save_dir, "information", unsuccessful_node_list_name)
    if os.path.exists(unsuccessful_node_list_path):
        unsuccessful_nodes = load4json(unsuccessful_node_list_path, [])
    else:
        with open(unsuccessful_node_list_path, "a"):
            os.utime(unsuccessful_node_list_path, None)
        unsuccessful_nodes = []

    try:
        for node in os.listdir(SaltAI_Web_Docs_path):
            node_path = os.path.join(SaltAI_Web_Docs_path, node)
            if os.path.isdir(node_path) and len(os.listdir(node_path)) > 0:
                for item in os.listdir(node_path):
                    if item == "Nodes":
                        sub_node_dir = os.path.join(node_path, item)
                        for sub_node in os.listdir(sub_node_dir):
                            sub_node_path = os.path.join(sub_node_dir, sub_node)
                            if sub_node_path not in successful_nodes:
                                try: 
                                    sub_node_name = os.path.splitext(sub_node)[0]
                                    rsp = ''
                                    rsp = get_data_from_siliconflow(sub_node_name, sub_node_path,
                                                                    model='alibaba/Qwen2-72B-Instruct',
                                                                    system_prompt=system_prompt2,
                                                                    template=template2)
                                    time.sleep(3)
                                    rsp_json = parse_json(rsp)
                                    save2json(rsp_json, os.path.join(save_dir, f"{node}+{sub_node_name}.json"))
                                    successful_nodes.append(sub_node_path)
                                    if sub_node_path in unsuccessful_nodes:
                                        unsuccessful_nodes.remove(sub_node_path)
                                    logger.info(f'Successfully extracting data from file: {sub_node_path}')
                                    # break
                                except Exception as e:
                                    logger.error(f'Failed to extract data from file: {sub_node_path}, error: {e}')
                                    if sub_node_path not in unsuccessful_nodes:
                                        unsuccessful_nodes.append(sub_node_path)
                    elif item == "index.md":
                        try: 
                            index_path = os.path.join(node_path, item)
                            if index_path not in successful_nodes:
                                rsp = ''
                                rsp = get_data_from_siliconflow(node, index_path,
                                                                model='alibaba/Qwen2-72B-Instruct',
                                                                system_prompt=system_prompt2_index,
                                                                template=template2_index)
                                time.sleep(3)
                                rsp_json = parse_json(rsp)
                                save2json(rsp_json, os.path.join(save_dir, f"{node}.json"))
                                successful_nodes.append(index_path)
                                if index_path in unsuccessful_nodes:
                                        unsuccessful_nodes.remove(index_path)
                                logger.info(f'Successfully extracting data from file: {index_path}')
                        except Exception as e:
                            logger.error(f'Failed to extract data from file: {index_path}, error: {e}')
                            if index_path not in unsuccessful_nodes:
                                unsuccessful_nodes.append(index_path)
                    else:
                        continue
    finally:
        save2json(successful_nodes, successful_node_list_path)
        save2json(unsuccessful_nodes, unsuccessful_node_list_path)


def generate_data_from_comfyui_nodes_docs(comfyui_nodes_docs_path: str = r"D:\git_github\self\ComfyChat\data\community_docs\repos\comfyui-nodes-docs\docs",
                                          save_dir: str = r"D:\git_github\self\ComfyChat\data\community_docs\messages\comfyui-nodes-docs",
                                          successful_node_list_name: str = "successful_node_list.json",
                                          unsuccessful_node_list_name: str = "unsuccessful_node_list.json") -> None:
    logger = create_logger("generate_data_from_comfyui_nodes_docs")

    successful_node_list_path = os.path.join(save_dir, "information", successful_node_list_name)
    if os.path.exists(successful_node_list_path):
        successful_nodes = load4json(successful_node_list_path, [])
    else:
        with open(successful_node_list_path, "a"):
            os.utime(successful_node_list_path, None)
        successful_nodes = []

    unsuccessful_node_list_path = os.path.join(save_dir, "information", unsuccessful_node_list_name)
    if os.path.exists(unsuccessful_node_list_path):
        unsuccessful_nodes = load4json(unsuccessful_node_list_path, [])
    else:
        with open(unsuccessful_node_list_path, "a"):
            os.utime(unsuccessful_node_list_path, None)
        unsuccessful_nodes = []

    try:
        for item in os.listdir(comfyui_nodes_docs_path):
            md_path = os.path.join(comfyui_nodes_docs_path, item)
            if not os.path.isdir(md_path):
                name, ext = extract_name_extension(item)
                if ext in ['.MD', '.MDX', '.md', '.mdx'] and md_path not in successful_nodes:
                    try:
                        save_path = os.path.join(save_dir, f"{name}.json")
                        if not os.path.exists(save_path):
                            rsp = ''
                            rsp = get_data_from_siliconflow(name, md_path,
                                                            model='alibaba/Qwen2-72B-Instruct',
                                                            system_prompt=system_prompt_zh,
                                                            template=template_zh)
                            time.sleep(3)
                            rsp_json = parse_json(rsp)
                            save2json(rsp_json, save_path)
                            successful_nodes.append(md_path)
                            if md_path in unsuccessful_nodes:
                                unsuccessful_nodes.remove(md_path)
                            logger.info(f'Successfully extracting data from file: {md_path}')
                        # break
                    except Exception as e:
                        logger.error(f'Failed to extract data from file: {md_path}, error: {e}')
                        if md_path not in unsuccessful_nodes:
                            unsuccessful_nodes.append(md_path)
    finally:
        save2json(successful_nodes, successful_node_list_path)
        save2json(unsuccessful_nodes, unsuccessful_node_list_path)


def generate_data_from_comflowy(comflowy_path: str = r"D:\git_github\self\ComfyChat\data\community_docs\repos\comflowy\pages",
                                save_dir: str = r"D:\git_github\self\ComfyChat\data\community_docs\messages\comflowy",
                                successful_node_list_name: str = "successful_node_list.json",
                                unsuccessful_node_list_name: str = "unsuccessful_node_list.json") -> None:
    logger = create_logger("generate_data_from_comflowy")

    successful_node_list_path = os.path.join(save_dir, "information", successful_node_list_name)
    if os.path.exists(successful_node_list_path):
        successful_nodes = load4json(successful_node_list_path, [])
    else:
        with open(successful_node_list_path, "a"):
            os.utime(successful_node_list_path, None)
        successful_nodes = []

    unsuccessful_node_list_path = os.path.join(save_dir, "information", unsuccessful_node_list_name)
    if os.path.exists(unsuccessful_node_list_path):
        unsuccessful_nodes = load4json(unsuccessful_node_list_path, [])
    else:
        with open(unsuccessful_node_list_path, "a"):
            os.utime(unsuccessful_node_list_path, None)
        unsuccessful_nodes = []
    
    try:
        for item in os.listdir(comflowy_path):
            md_dir = os.path.join(comflowy_path, item)
            if os.path.isdir(md_dir):
                for md in os.listdir(md_dir):
                    name, ext = extract_name_extension(md)
                    md_path = os.path.join(md_dir, md)
                    if ext in ['.MD', '.MDX', '.md', '.mdx'] and md_path not in successful_nodes:
                        try:
                            save_path = os.path.join(save_dir, f"{name}.json")
                            _, flag = extract_name_extension(name)
                            if flag == ".en-US":
                                system_prompt = system_prompt2_index
                                template = template2_index
                            elif flag == ".zh-CN":
                                system_prompt = system_prompt_zh
                                template = template_zh
                            else:
                                raise ValueError("文件标识错误")
                            rsp = ''
                            rsp = get_data_from_siliconflow(name, md_path,
                                                            model='alibaba/Qwen2-72B-Instruct',
                                                            system_prompt=system_prompt,
                                                            template=template)
                            time.sleep(3)
                            rsp_json = parse_json(rsp)
                            save2json(rsp_json, save_path)
                            successful_nodes.append(md_path)
                            if md_path in unsuccessful_nodes:
                                unsuccessful_nodes.remove(md_path)
                            logger.info(f'Successfully extracting data from file: {md_path}')
                        except Exception as e:
                            logger.error(f'Failed to extract data from file: {md_path}, error: {e}')
                            if md_path not in unsuccessful_nodes:
                                unsuccessful_nodes.append(md_path)
    finally:
        save2json(successful_nodes, successful_node_list_path)
        save2json(unsuccessful_nodes, unsuccessful_node_list_path)


def constrcut_messages_from_community_docs(base_dir: str, save_path: str) -> None:
    messages = []

    for item in os.listdir(base_dir):
        if item.endswith('.json'):
            try:
                json_path = os.path.join(base_dir, item)
                json_datas = load4json(json_path)
                if isinstance(json_datas, dict):
                    datas = []
                    for k in json_datas:
                        datas += json_datas[k]
                else:
                    datas = json_datas
                for data in datas:
                    message = construct_single_messages(data['question'], data['answer'])
                    messages.append(message)
            except Exception as e:
                print(json_path)

    save2json(messages, save_path)


def constrcut_messages_from_comflowy() -> None:
    en_messages = []
    zh_messages = []
    base_dir = r"D:\git_github\self\ComfyChat\data\community_docs\messages\comflowy"

    for item in os.listdir(base_dir):
        if item.endswith('.json'):
            try:
                json_path = os.path.join(base_dir, item)
                json_datas = load4json(json_path)
                if isinstance(json_datas, dict):
                    datas = []
                    for k in json_datas:
                        datas += json_datas[k]
                else:
                    datas = json_datas
                for data in datas:
                    message = construct_single_messages(data['question'], data['answer'])
                    if item.endswith(".en-US.json"):
                        en_messages.append(message)
                    elif item.endswith(".zh-CN.json"):
                        zh_messages.append(message)
            except Exception as e:
                print(json_path)

    save2json(en_messages, r"D:\git_github\self\ComfyChat\data\message_jsons\v2\comflowy_en.json")
    save2json(zh_messages, r"D:\git_github\self\ComfyChat\data\message_jsons\v2\comflowy_zh.json")


def combine_messages(message_paths: List[str], save_path: str, shuffle: bool = False) -> None:
    messages = []
    for path in message_paths:
        messages.extend(load4json(path))

    if shuffle:
        random.seed(42)
        random.shuffle(messages)
    print(f"nums of conversations: {len(messages)}")
    save2json(messages, save_path)


if __name__=='__main__':
    # md2txt()
    # construct_data_from_custom_node_list(together=True, seve_path='/root/code/ComfyChat/data/comfyui_node_data_together.json')
    # alpaca_modify('/root/code/ComfyChat/data/alpaca_gpt4_data_zh.json', '/root/code/ComfyChat/data/alpaca_gpt4_data_zh_modification.json')
    
    # construct_data_zh_from_custom_node_list()

    # eng_text = "The author of FreeU_Advanced is WASasquatch, This custom node provides advanced settings for FreeU."
    # eng2zh_deepseek(eng_text)

    # get_data_from_openrouter('ComfyUI_Fictiverse', '/root/code/ComfyChat/data/custom_nodes_mds/ComfyUI_Fictiverse/README.md')

    # constrcut_data_from_md()

    # print(get_data_from_deepseek('ComfyUI_Fictiverse', '/root/code/ComfyChat/data/custom_nodes_mds/ComfyUI_Fictiverse/README.md'))

    # parse_data_from_md_json()

    # questions = [
    #             "What is the purpose of the Cozy Sampler Options node?",
    #             "Who created the Cozy Sampler Options node for ComfyUI?",
    #             "What is the ComfyUI Sampler node used for?",
    #             "Can you provide an image of the ComfyUI Sampler Options node?"
    #         ]
    # answers = [
    #             "The Cozy Sampler Options node is a simple node designed to generate a list of options for the ComfyUI Sampler node.",
    #             "Cozy Sampler Options was created by the CozyMantis squad.",
    #             "The ComfyUI Sampler node is utilized to provide a set of options for various tasks within the ComfyUI interface.",
    #             "Yes, an image of the ComfyUI Sampler Options node can be found at ./assets/node.png."
    #         ]
    # node_name = 'cozy-utils-comfyui-nodes'
    # semi_automatic_for_one_node1(node_name, questions, answers)

    # qa = {
    #         "content": [
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "Is the ComfyUI-Manager extension still working after the SD XL update?",
    #                 "answer": "No, the workaround used to patch the hardcoded transformer model from the HuggingFace library no longer works after the SD XL update."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What is the contribution of the ComfyUI-Manager extension to Stable Diffusion?",
    #                 "answer": "At present, the extension is not contributing significantly enough to justify additional development time."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "How does the directional prompt attention extension affect the CLIP and SD parts of the framework?",
    #                 "answer": "The extension only affects the CLIP part of the framework, but since the SD part is conditioned on a summarized representation of the prompt, the SD part still sees all inputs, making it difficult for the method to work consistently."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What is Directional Prompt Attention in the context of ComfyUI?",
    #                 "answer": "Directional Prompt Attention is an attempt to limit the impact of contextual words or parts of the prompt on subsequent or irrelevant parts of the prompt."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What is the purpose of the causal attention mask in the standard transformer implementation?",
    #                 "answer": "The causal attention mask prevents the current tokens from attending to future tokens, which is useful for language models that are trained to predict the next word."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "How is causal attention masking implemented within CLIP transformer models?",
    #                 "answer": "The standard CLIP transformer has a built-in causal attention mask that masks out future tokens from the current tokens' attention."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What does the 'ComfyUI-Manager' extension implement regarding attention masks?",
    #                 "answer": "The extension allows the transformer to apply attention only on certain tokens in the prompt to limit the effect of contextual words or parts of the prompt on subsequent or irrelevant parts of the prompt."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "How does the user specify relationships in the prompt using the ComfyUI-Manager extension?",
    #                 "answer": "The user specifies relationships in the prompt using parentheses, `<`, and `>`."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What is the 'CLIP Directional Prompt Attention Encode' node used for in ComfyUI?",
    #                 "answer": "This node allows users to use `>` and `<` in the prompt to denote relationships between words or parts of the prompt."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "Where can the 'CLIP Directional Prompt Attention Encode' node be found in ComfyUI?",
    #                 "answer": "This node can be found under `conditioning` in ComfyUI."
    #             },
    #             {
    #                 "subject": "ComfyUI-Manager",
    #                 "question": "What additional packages are required to use this extension in ComfyUI?",
    #                 "answer": "You will need `scikit-learn` and `matplotlib` installed in your ComfyUI environment to use this extension."
    #             }
    #         ]
    #     }
    # node_name = 'ComfyUI-Manager'
    # semi_automatic_for_one_node2(node_name, qa)

    # construct_data("/root/code/ComfyChat/data/comfyui_data_v1.json")

    # check_messages_json()

    # comfyui_data = load4json('/root/code/ComfyChat/data/comfyui_data_v1.json')
    # for v in comfyui_data:
    #     if isinstance(v, dict) and "messages" in v and len(v["messages"]) == 2:
    #         d1 = v["messages"][0]
    #         d2 = v["messages"][1]
    #         if ("role" in d1 and d1["role"] and "content" in d1 and d1["content"]) and ("role" in d2 and d2["role"] and "content" in d2 and d2["content"]):
    #             continue
    #         else:
    #             print(v)
    #     else:
    #         print(v)


    # from datasets import load_dataset
    # comfyui_data = load_dataset("json", data_file='/root/code/ComfyChat/data/comfyui_node_data.json')
    # comfyui_data = comfyui_data['json']
    # print(len(comfyui_data))

    # translate_final2zh()
    
    # ans = eng2zh_chat2api(eng_text='America is fucking shit')
    # print(ans)

    # temp = get_data_from_siliconflow('CropLatent',
    #                              r'D:\\git_github\\self\\ComfyChat\\data\\community_docs\\repos\\ComfyUI-docs\\docs\\Core Nodes\\Latent\\transform\\CropLatent.md',
    #                              model='alibaba/Qwen2-72B-Instruct',
    #                              system_prompt=system_prompt2_index, template=template2_index)
    # print(temp)
    # temp = parse_json(temp)
    # print(temp)
    # save2json(temp, r"D:\git_github\self\ComfyChat\data\community_docs\messages\ComfyUI-docs\CropLatent.json")

    # generate_data_from_comfyui_docs()

    # generate_data_from_SaltAI_Web_Docs()

    # generate_data_from_comfyui_nodes_docs()

    # test = "preparation-for-study.en-US.mdx"
    # name, ext = extract_name_extension(test)
    # print(name, ext)
    # name, ext = extract_name_extension(name)
    # print(name, ext)

    # generate_data_from_comflowy()

    # base_dir = r"D:\git_github\self\ComfyChat\data\community_docs\messages\SaltAI-Web-Docs"
    # save_dir = r"D:\git_github\self\ComfyChat\data\message_jsons\v2\SaltAI-Web-Docs.json"
    # constrcut_messages_from_community_docs(base_dir, save_dir)

    # constrcut_messages_from_comflowy()

    v2_paths = [r"D:\git_github\self\ComfyChat\data\message_jsons\v1\comfyui_data_v1.json",
                r"D:\git_github\self\ComfyChat\data\message_jsons\v2\community_en.json"]
    save_path = r"D:\git_github\self\ComfyChat\data\message_jsons\v2\comfyui_data_v2_1.json"

    combine_messages(v2_paths, save_path, shuffle=True)

    # messages = load4json(r"D:\git_github\self\ComfyChat\data\message_jsons\v1\comfyui_data_v1.json")
    # print(len(messages))