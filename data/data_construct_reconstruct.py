#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2024/08/19 23:45:56
@Author  :   zzfive 
@Desc    :   None
'''
import os
import time
import copy
import shutil
import random
from typing import Any, Dict, List
from datetime import datetime, timedelta

import requests
from git import Repo
from openai import OpenAI

import config
from prompt_templates import system_prompt1, template1, system_prompt2, template2, system_prompt_zh, template_zh, system_prompt2_index, template2_index
from get_custom_node_markdowns import extract_md_files_from_local_repo
from utils import create_logger, get_data_from_url, load4json, save2json, extract_name_extension, parse_json


logger = create_logger("data_construct")


# 此类限制每分钟请求次数
class RPM:

    def __init__(self, rpm: int = 30) -> None:
        self.rpm = rpm
        self.record = {'slot': self.get_minute_slot(), 'counter': 0}  # 分钟槽和计数器

    # 获取分钟槽，是从午夜开始算起，可以唯一标识一天中的每一分钟
    def get_minute_slot(self) -> int:
        current_time = time.time()
        dt_object = datetime.fromtimestamp(current_time)
        total_minutes_since_midnight = dt_object.hour * 60 + dt_object.minute
        return total_minutes_since_midnight

    def wait(self) -> None:
        current = time.time()
        dt_object = datetime.fromtimestamp(current)
        minute_slot = self.get_minute_slot()  # 当前时间的分钟槽

        if self.record['slot'] == minute_slot:  # 如果当前分钟槽与记录中分钟槽一致
            # check RPM exceed
            if self.record['counter'] >= self.rpm:  # 计数器已大于等于RPM限制
                # wait until next minute
                next_minute = dt_object.replace(second=0, microsecond=0) + timedelta(minutes=1)
                _next = next_minute.timestamp()
                sleep_time = abs(_next - current)
                time.sleep(sleep_time)  # 等待到下一分钟

                self.record = {'slot': self.get_minute_slot(), 'counter': 0}  # 记录的分钟槽和计数器重置
        else:
            self.record = {'slot': self.get_minute_slot(), 'counter': 0}  # 如果分钟槽不一致，重置记录
        self.record['counter'] += 1
        logger.debug(self.record)


# 支持多种LLMs接口进行英文翻译为中文和基于comfyui相关文档生成messages类型数据
class LLMApiGenerator:
    def __init__(self, rpm: int = 10) -> None:
        self.rpm = RPM(rpm)
        self.backend_settings = {
            "kimi":{"base_url": "https://api.moonshot.cn/v1",
                    "defualt_model": "moonshot-v1-8k",
                    "api_key": config.MOONSHOT_API_KEY},
            "deepseek":{"base_url": "https://api.deepseek.com/v1",
                        "defualt_model": "deepseek-chat",
                        "api_key": config.DEEPSEEK_API_KEY},
            "openrouter":{"base_url": "https://openrouter.ai/api/v1",
                          "defualt_model": "nousresearch/hermes-3-llama-3.1-405b",
                          "api_key": config.OPENROUTER_API_KEY},
            "siliconflow":{"base_url": "https://api.siliconflow.cn/v1/chat/completions",
                           "defualt_model": "internlm/internlm2_5-7b-chat",
                           "api_key": config.SILICONFLOW_API_KEY},
            "chat2api":{"base_url": "http://127.0.0.1:5005/v1/chat/completions",
                        "defualt_model": "gpt-3.5-turb",
                        "api_key": config.OPENAI_ACCESS_TOKEN},
        }

    def eng2zh_openai(self, eng_text: str, base_url: str, api_key: str, model: str) -> str:
        self.rpm.wait()
        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是英汉翻译大师。 请将用户输入的英文文本准确翻译成中文。 一些专有名词可以保留而无需翻译。"},
                {"role": "user", "content": f"将以下文字翻译成中文，不要添加任何无关内容：{eng_text}"}
            ],
        )

        ans = completion.choices[0].message.content
        return ans

    def eng2zh_requests(self, eng_text: str, base_url: str, api_key: str, model: str) -> str:
        self.rpm.wait()
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
            "authorization": f"Bearer {api_key}"
        }

        response = requests.post(base_url, json=payload, headers=headers).json()
        ans = response['choices'][0]['message']['content']
        
        return ans

    def eng2zh_llm(self, eng_text: str, backend: str = "kimi", api_key: str = None, base_url: str = None, model: str = None) -> str:
        api_key = self.backend_settings[backend]["api_key"] if api_key is None else api_key
        base_url = self.backend_settings[backend]["base_url"] if base_url is None else base_url
        model = self.backend_settings[backend]["defualt_model"] if model is None else model

        if backend in ["kimi", "deepseek", "openrouter"]:
            ans = self.eng2zh_openai(eng_text, base_url, api_key, model)
        elif backend in ["siliconflow", "chat2api"]:
            ans = self.eng2zh_requests(eng_text, base_url, api_key, model)
        else:
            raise ValueError(f"{backend}不支持")
        
        return ans
    
    def messages_generate_openai(self, subject: str, file_path: str, base_url: str, api_key: str, model: str,
                                 system_prompt: str = system_prompt1, template: str = template1) -> str:
        self.rpm.wait()
        client = OpenAI(api_key=api_key, base_url=base_url)

        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template.format(subject, file_content)},
        ],
        temperature=1.0,
        )
        return completion.choices[0].message.content

    def messages_generate_requests(self, subject: str, file_path: str, base_url: str, api_key: str, model: str,
                                   system_prompt: str = system_prompt1, template: str = template1) -> str:
        self.rpm.wait()
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": template.format(subject, file_content)}
            ]
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }

        response = requests.post(base_url, json=payload, headers=headers)

        ans = response.json()
        ans = ans['choices'][0]['message']['content']
        return ans

    # TODO 目前生成的都是单一对话问题，后续可以尝试生成连续对话问题
    def messages_generate_llm(self, subject: str, file_path: str, backend: str = "kimi", base_url: str = None, api_key: str = None,
                              model: str = None, system_prompt: str = system_prompt1, template: str = template1) -> str:
        api_key = self.backend_settings[backend]["api_key"] if api_key is None else api_key
        base_url = self.backend_settings[backend]["base_url"] if base_url is None else base_url
        model = self.backend_settings[backend]["defualt_model"] if model is None else model

        if backend in ["kimi", "deepseek", "openrouter"]:
            ans = self.messages_generate_openai(subject, file_path, base_url, api_key, model, system_prompt, template)
        elif backend in ["siliconflow", "chat2api"]:
            ans = self.messages_generate_requests(subject, file_path, base_url, api_key, model, system_prompt, template)
        else:
            raise ValueError(f"{backend}不支持")
        
        return ans
    

# 基于comfyui-manager构建messages的pipeline
class DataCollectAndMessagesGeneratePipelineWithComfyuiManager:
    custom_node_list_json_url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"  # 自定义节点列表
    custom_node_map_json_url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"  # 包含自定义节点的子节点信息
    github_stats_json_url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/github-stats.json"  # 自定义节点的更新信息
    model_list_json_url = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"  # 常用模型的下载地址等信息

    local_custom_node_infos_path = "/root/code/ComfyChat/data/custom_node_infos.json"

    def __init__(self) -> None:
        self.remote_custom_node_list = get_data_from_url(self.custom_node_list_json_url)['custom_nodes']
        self.remote_github_stats = get_data_from_url(self.github_stats_json_url)

        if os.path.exists(self.local_custom_node_infos_path):
            self.local_custom_node_infos = load4json(self.local_custom_node_infos_path, {})
        else:
            self.local_custom_node_infos = {}
        
        if self.local_custom_node_infos == {}:
            self.touch_infos()

        self.llm_generator = LLMApiGenerator(100)

    def touch_infos(self, local_custom_node_repos_dir: str = "/root/code/ComfyChat/data/custom_nodes_repos",
                    local_custom_node_mds_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds",
                    local_custom_node_jsons_dir: str = "/root/code/ComfyChat/data/custom_nodes_jsons") -> None:
        for node_info in self.remote_custom_node_list:
            node_url = node_info['reference']
            node_name = node_url.split("/")[-1]
            local_node_repo_path = os.path.join(local_custom_node_repos_dir, node_name)
            local_node_md_path = os.path.join(local_custom_node_mds_dir, node_name)
            local_node_json_path = os.path.join(local_custom_node_jsons_dir, node_name)

            if os.path.exists(local_node_repo_path) and len(os.listdir(local_node_repo_path)) > 0:  # 拉取了当前节点的github项目
                if node_url in self.local_custom_node_infos:
                    if self.local_custom_node_infos[node_url].get("local_last_update", None) is None:
                        self.local_custom_node_infos[node_url]["local_last_update"] = "2024-05-25 00:00:00"
                else:
                    self.local_custom_node_infos[node_url] = {}
                    self.local_custom_node_infos[node_url]["local_last_update"] = "2024-05-25 00:00:00"
                self.local_custom_node_infos[node_url]["repo_cloned"] = True
            
            if os.path.exists(local_node_md_path) and len(os.listdir(local_node_md_path)) > 0:  # 从拉取的github项目中过滤出了md文档
                if node_url in self.local_custom_node_infos:
                    if self.local_custom_node_infos[node_url].get("md_last_update", None) is None:
                        self.local_custom_node_infos[node_url]["md_last_update"] = "2024-05-25 00:00:00"
                else:
                    self.local_custom_node_infos[node_url] = {}
                    self.local_custom_node_infos[node_url]["md_last_update"] = "2024-05-25 00:00:00"
                self.local_custom_node_infos[node_url]["repo_md"] = True

            if os.path.exists(local_node_json_path) and len(os.listdir(local_node_json_path)) > 0:  # 针对过滤后的md文档使用llm生成了问答json数据
                if node_url in self.local_custom_node_infos:
                    if self.local_custom_node_infos[node_url].get("json_last_update", None) is None:
                        self.local_custom_node_infos[node_url]["json_last_update"] = "2024-05-25 00:00:00"
                else:
                    self.local_custom_node_infos[node_url] = {}
                    self.local_custom_node_infos[node_url]["json_last_update"] = "2024-05-25 00:00:00"
                self.local_custom_node_infos[node_url]["repo_json"] = True
            
            if node_url in self.local_custom_node_infos:
                if "local_last_update" not in self.local_custom_node_infos[node_url]:
                    self.local_custom_node_infos[node_url]["local_last_update"] = None
                    self.local_custom_node_infos[node_url]["repo_cloned"] = False

                if "md_last_update" not in self.local_custom_node_infos[node_url]:
                    self.local_custom_node_infos[node_url]["md_last_update"] = None
                    self.local_custom_node_infos[node_url]["repo_md"] = False

                if "json_last_update" not in self.local_custom_node_infos[node_url]:
                    self.local_custom_node_infos[node_url]["json_last_update"] = None
                    self.local_custom_node_infos[node_url]["repo_json"] = False
            else:
                self.local_custom_node_infos[node_url] = {}
                self.local_custom_node_infos[node_url]["local_last_update"] = None
                self.local_custom_node_infos[node_url]["repo_cloned"] = False
                self.local_custom_node_infos[node_url]["md_last_update"] = None
                self.local_custom_node_infos[node_url]["repo_md"] = False
                self.local_custom_node_infos[node_url]["json_last_update"] = None
                self.local_custom_node_infos[node_url]["repo_json"] = False

            self.local_custom_node_infos[node_url]["json_version"] = 1
            self.local_custom_node_infos[node_url]["remote_last_update"] = self.remote_github_stats[node_url].get("last_update", None) if node_url in self.remote_github_stats else None

            self.local_custom_node_infos[node_url]["successful_files"] = []
            self.local_custom_node_infos[node_url]["unsuccessful_files"] = []


    # 对当前self.local_custom_node_infos中的所有节点仓库进行更新
    def refresh_all_repos(self, local_custom_node_repos_dir: str = "/root/code/ComfyChat/data/custom_nodes_repos") -> None:
        try:
            for k, v in self.local_custom_node_infos.items():
                node_name = k.split("/")[-1]
                local_node_repo_path = os.path.join(local_custom_node_repos_dir, node_name)
                try:
                    if (not v["repo_cloned"] or v["local_last_update"] is None) and not os.path.exists(local_node_repo_path):
                        repo = Repo.clone_from(k, local_node_repo_path)
                        v["local_last_update"] = v["remote_last_update"]
                        v["repo_cloned"] = True
                        v["remote_last_update"] = self.remote_github_stats[k].get("last_update", None) if k in self.remote_github_stats else None
                        if v["remote_last_update"] is None:
                            v["remote_last_update"] = repo.head.commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    if v["local_last_update"] is not None and v["remote_last_update"] is not None and v["local_last_update"] < v["remote_last_update"]:  # 以获取的远程时间进行比较，判断是否更新本地仓库
                        repo = Repo(local_node_repo_path)  # 打开本地仓库
                        origin = repo.remotes.origin  # 获取远程仓库
                        origin.pull()  # 从远程仓库拉取最新代码
                        v["local_last_update"] = v["remote_last_update"]
                        v["repo_cloned"] = True
                        v["remote_last_update"] = self.remote_github_stats[k].get("last_update", None) if k in self.remote_github_stats else None
                        if v["remote_last_update"] is None:
                            v["remote_last_update"] = repo.head.commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logger.error(f"{k} refresh failure: {e}")
        finally:
            self.save_infos()

    # 对一个节点进行更新，如果在节点仓库之前未拉取，会直接拉取
    def refresh_one_repo(self, node_url: str, local_custom_node_repos_dir: str = "/root/code/ComfyChat/data/custom_nodes_repos") -> None:
        node_name = node_url.split("/")[-1]
        local_node_repo_path = os.path.join(local_custom_node_repos_dir, node_name)
        try:
            if node_url in self.local_custom_node_infos:
                repo = Repo(local_node_repo_path)  # 打开本地仓库
                origin = repo.remotes.origin  # 获取远程仓库
                origin.pull()  # 从远程仓库拉取最新代码
            else:
                self.local_custom_node_infos[node_url] = {}
                Repo.clone_from(node_url, local_node_repo_path)
                repo = Repo(local_node_repo_path)

            self.local_custom_node_infos[node_url]["repo_cloned"] = True
            latest_commit = repo.head.commit
            latest_commit_time = latest_commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
            self.local_custom_node_infos[node_url]["local_last_update"] = latest_commit_time
            self.local_custom_node_infos[node_url]["remote_last_update"] = self.remote_github_stats[node_url].get("last_update", None) if node_url in self.remote_github_stats else None
        except Exception as e:
            logger.error(f"{node_url} refresh failure: {e}")
            self.local_custom_node_infos[node_url]["repo_cloned"] = False

    # 对当前self.local_custom_node_infos中的所有节点仓库中包含的md文档刷新
    def refresh_all_mds(self, local_custom_node_repos_dir: str = "/root/code/ComfyChat/data/custom_nodes_repos",
                        local_custom_node_mds_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
        try:
            for k, v in self.local_custom_node_infos.items():
                if not v["repo_md"] or v["md_last_update"] is None or v["md_last_update"] < v["local_last_update"]:
                    node_name = k.split("/")[-1]
                    local_node_repo_path = os.path.join(local_custom_node_repos_dir, node_name)
                    local_node_md_path = os.path.join(local_custom_node_mds_dir, node_name)
                    try:
                        if os.path.exists(local_node_md_path):
                            shutil.rmtree(local_node_md_path)
                        extract_md_files_from_local_repo(local_node_repo_path, local_node_md_path)
                        v["md_last_update"] = v["local_last_update"]
                        v["repo_md"] = True
                    except Exception as e:
                        logger.error(f"{k} MD refresh failure: {e}")
        finally:
            self.save_infos()

    def refresh_one_md(self, node_url: str, local_custom_node_repos_dir: str = "/root/code/ComfyChat/data/custom_nodes_repos",
                       local_custom_node_mds_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds") -> None:
        node_name = node_url.split("/")[-1]
        local_node_repo_path = os.path.join(local_custom_node_repos_dir, node_name)
        local_node_md_path = os.path.join(local_custom_node_mds_dir, node_name)
        try:
            if node_url in self.local_custom_node_infos and self.local_custom_node_infos[node_url]["repo_md"]:
                if os.path.exists(local_node_md_path):
                    shutil.rmtree(local_node_md_path)
            else:
                self.refresh_one_repo(node_url)

            extract_md_files_from_local_repo(local_node_repo_path, local_node_md_path)
            self.local_custom_node_infos[node_url]["md_last_update"] = self.local_custom_node_infos[node_url]["local_last_update"]
            self.local_custom_node_infos[node_url]["repo_md"] = True
        except Exception as e:
            logger.error(f"{node_url} MD refresh failure: {e}")
            self.local_custom_node_infos[node_url]["repo_md"] = False

    def refresh_all_jsons(self, version: int) -> None:
        for k, v in self.local_custom_node_infos.items():
            if not v["repo_json"] or v["json_last_update"] is None or v["json_last_update"] < v["md_last_update"]:
                self.refresh_one_json(k, version)
        self.save_infos()

    def refresh_one_json(self, node_url: str, version: int, lang: str = "en",
                         local_custom_node_mds_dir: str = "/root/code/ComfyChat/data/custom_nodes_mds",
                         local_custom_node_jsons_dir: str = "/root/code/ComfyChat/data/custom_nodes_jsons") -> None:
        try:
            if node_url not in self.local_custom_node_infos or not self.local_custom_node_infos[node_url]["repo_cloned"]:
                self.refresh_one_repo(node_url)
            if not self.local_custom_node_infos[node_url]["repo_md"]:
                self.refresh_one_md(node_url)

            node_name = node_url.split("/")[-1]
            local_node_md_path = os.path.join(local_custom_node_mds_dir, node_name)
            local_node_json_path = os.path.join(local_custom_node_jsons_dir, node_name)
            if not os.path.exists(local_node_json_path):
                    os.mkdir(local_node_json_path)
            local_node_json_version_path = os.path.join(local_node_json_path, str(version))
            if not os.path.exists(local_node_json_version_path):
                os.mkdir(local_node_json_version_path)

            for item in os.listdir(local_node_md_path):
                md_path = os.path.join(local_node_md_path, item)
                if md_path not in self.local_custom_node_infos[node_url]["successful_files"]:
                    try:
                        md_name, _ = os.path.splitext(item)
                        if lang == "en":
                            rsp = self.llm_generator.messages_generate_llm(item, md_path)
                        if lang == "zh":
                            rsp = self.llm_generator.messages_generate_llm(item, md_path,
                                                                        system_prompt=system_prompt_zh,
                                                                        template=template_zh)
                        rsp_json = parse_json(rsp)
                        json_path = os.path.join(local_node_json_version_path, f"{md_name}{'_zh' if lang=='zh' else ''}.json")
                        if os.path.exists(json_path):
                            old_json = load4json(json_path, [])
                            old_json.extend(rsp_json)
                        else:
                            old_json = rsp_json
                        save2json(old_json, json_path)
                        self.local_custom_node_infos[node_url]["successful_files"].append(md_path)
                        if md_path in self.local_custom_node_infos[node_url]["unsuccessful_files"]:
                            self.local_custom_node_infos[node_url]["unsuccessful_files"].remove(md_path)
                        self.local_custom_node_infos[node_url]["json_last_update"] = self.local_custom_node_infos[node_url]["md_last_update"]
                        self.local_custom_node_infos[node_url]["repo_json"] = True
                    except Exception as e:
                        logger.error(f"Json of {md_path} refresh failure: {e}")
                        self.local_custom_node_infos[node_url]["unsuccessful_files"].append(md_path)
        except Exception as e:
            logger.error(f"{node_url} Json refresh failure: {e}")
            self.local_custom_node_infos[node_url]["repo_json"] = False
        finally:
            self.local_custom_node_infos[node_url]["json_version"] = version

    def construct_single_messages(self, user_content: str, assistant_content: str) -> Dict[str, List[Dict[str, str]]]:
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

    # llm生成的json数据不一定完全符合要求的结构，需要校验和解析，并记录结构有误的文件，便于手动调整
    def check_parse_jsons_single(self, version: int, lang: str = "en",
                                 local_custom_node_jsons_dir: str = "/root/code/ComfyChat/data/custom_nodes_jsons"):
        for node in os.listdir(local_custom_node_jsons_dir):
            node_version_dir = os.path.join(local_custom_node_jsons_dir, node, str(version))
            json_file_name = "final_zh.json" if lang == "zh" else "final.json"
            if os.path.exists(node_version_dir) and os.path.isdir(node_version_dir) and len(os.listdir(node_version_dir)) > 0 and json_file_name not in os.listdir(node_version_dir):
                messages = []
                try:
                    for item in os.path.isdir(node_version_dir):
                        if item.endswith('.json'):
                            try:
                                qa_path = os.path.join(node_version_dir, item)
                                qa = load4json(qa_path)
                                if isinstance(qa, list) and isinstance(qa[0], dict):
                                    list_dict = qa
                                    for v in list_dict:
                                        user_content = v.get('input', None) or v.get('question', None) or v.get('Question', None) or v.get('question_text', None) or v.get('prompt', None)
                                        assistant_content = v.get('output', None) or v.get('answer', None) or v.get('Answer', None) or v.get('answer_text', None) or v.get('completion', None)
                                        if user_content is not None and assistant_content is not None:
                                            messages.append(self.construct_single_messages(user_content, assistant_content))
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
                                                messages.append(self.construct_single_messages(user_content, assistant_content))
                                            else:
                                                logger.info(f"{v}")
                                    elif isinstance(qa[keys[0]], str):
                                        if len(keys) % 2 == 0:
                                            for i in range(0, len(keys), 2):
                                                messages.append(self.construct_single_messages(qa[keys[i]], qa[keys[i+1]]))
                                        else:
                                            raise ValueError(f'问答数据个数为基数')
                            except Exception as e:
                                logger.error(f"Error happened: {qa_path}, error: {e}")
                                pass
                finally:
                    if messages:
                        save2json(messages, os.path.join(node_version_dir, json_file_name))
                        logger.info(f"{os.path.join(node_version_dir, json_file_name)} saved \n")

    def constrcut_final_messages(self, version: int, save_path: str, lang: str = "en", kind: str = "single", shuffle: bool = False,
                                 seed: int = 42, local_custom_node_jsons_dir: str = "/root/code/ComfyChat/data/custom_nodes_jsons"):
        messages = []
        for node in os.listdir(local_custom_node_jsons_dir):
            node_version_dir = os.path.join(local_custom_node_jsons_dir, node, str(version))
            if kind == "single":
                json_file_name = "final_zh.json" if lang == "zh" else "final.json"
            else:
                json_file_name = ""
            node_final_json_path = os.path.join(node_version_dir, json_file_name)
            if os.path.exists(node_final_json_path):
                messages.extend(load4json(node_final_json_path))
        
        if shuffle:
            random.seed(seed)
            random.shuffle(messages)
        print(f"nums of conversations: {len(messages)}")
        save2json(messages, save_path)

    def save_infos(self):
        try:
            save2json(self.local_custom_node_infos,
                      self.local_custom_node_infos_path)
        except Exception as e:
            print(str(e))
            print(self.local_custom_node_infos)
            logger.error(self.local_custom_node_infos)
            raise ValueError("各自定义节点处理信息保存失败")


# TODO 简化当前对四个开源社区的数据提炼过程
class DataCollectAndMessagesGeneratePipelineWithCommunityProject:
    community_projects_infos = {
        "ComfyUI-docs": {
            "url": "https://github.com/BlenderNeko/ComfyUI-docs",
            "resources_dir": "docs\Core Nodes"
        },
        "SaltAI-Web-Docs": {
            "url": "https://github.com/get-salt-AI/SaltAI-Web-Docs",
            "resources_dir": "docs\md"
        },
        "comfyui-nodes-docs": {
            "url": "https://github.com/CavinHuang/comfyui-nodes-docs",
            "resources_dir": "docs"
        },
        "comflowy": {
            "url": "https://github.com/6174/comflowy",
            "resources_dir": "pages"
        }
    }

    local_community_project_infos_path = "/root/code/ComfyChat/data/community_project_infos.json"
    
    def __init__(self) -> None:
        if os.path.exists(self.local_community_project_infos_path):
            self.local_community_project_infos = load4json(self.local_community_project_infos_path, {})
        else:
            self.touch_infos()
        
        self.llm_generator = LLMApiGenerator(100)

    def touch_infos(self, repos_dir: str = "/root/code/ComfyChat/data/community_docs/repos",
                    jsons_dir: str = "/root/code/ComfyChat/data/community_docs/messages") -> None:
        for project_name in self.community_projects_infos:
            if project_name not in self.local_community_project_infos:
                self.local_community_project_infos[project_name] = {}

            local_repo_dir = os.path.join(repos_dir, project_name)
            local_resources_dir = os.path.join(local_repo_dir, self.community_projects_infos[project_name]["resources_dir"])
            if not os.path.exists(local_repo_dir) or not os.path.exists(local_resources_dir):
                self.local_community_project_infos[project_name]["valid_repo"] = False
            else:
                self.local_community_project_infos[project_name]["valid_repo"] = True
            if self.local_community_project_infos[project_name].get("local_last_update", None) is None:
                self.local_community_project_infos[project_name]["local_last_update"] = "2024-06-27 00:00:00"

            local_json_dir = os.path.join(jsons_dir, project_name)
            if os.path.exists(local_json_dir) and len(os.listdir(local_json_dir)) > 0:
                self.local_community_project_infos[project_name]["valid_json"] = True
            else:
                self.local_community_project_infos[project_name]["valid_json"] = False
            if self.local_community_project_infos[project_name].get("json_last_update", None) is None:
                self.local_community_project_infos[project_name]["json_last_update"] = "2024-06-27 00:00:00"

            self.local_community_project_infos[project_name]["successful_nodes"] = []
            self.local_community_project_infos[project_name]["unsuccessful_nodes"] = []

    def refresh_project_json(self, project_name: str, version: int, pull: bool= False, again: bool = False,
                             repos_dir: str = "/root/code/ComfyChat/data/community_docs/repos",
                             jsons_dir: str = "/root/code/ComfyChat/data/custom_nodes_jsons") -> None:
        try:
            if project_name not in self.community_projects_infos:
                raise ValueError(f"{project_name}项目暂未支持")

            if not os.path.join(jsons_dir, "ComfyUI-docs", str(version)) and version == 1:
                raise ValueError("version为1的路径不存在时，version要大于1")

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            local_repo_dir = os.path.join(repos_dir, project_name)
            local_jsons_dir = os.path.join(jsons_dir, project_name, str(version))
            if not os.path.exists(local_jsons_dir):
                os.mkdir(local_jsons_dir)
            local_resources_dir = os.path.join(local_repo_dir, self.community_projects_infos[project_name]["resources_dir"])

            if not self.local_community_project_infos[project_name]["valid_repo"]:  # 项目没clone到本地或主要包含有效文档的路径不存在
                try:
                    if os.path.exists(local_repo_dir):
                        shutil.rmtree(local_repo_dir)
                    Repo.clone_from(self.community_projects_infos[project_name]["url"], local_repo_dir)
                    self.local_community_project_infos[project_name]["valid_repo"] = True  # 只从头拉取项目，不做其他操作
                except Exception as e:
                    logger.error(f"{project_name}仓库更新失败, error {e}")
                    raise ValueError(f"{project_name}仓库更新失败")
            else:  # 本地的项目仓库完整，有效文档路径存在
                if pull:
                    try:
                        repo = Repo(local_repo_dir)
                        old_commit = repo.head.commit
                        origin = repo.remotes.origin
                        origin.pull()
                        new_commit = repo.head.commit
                    except Exception as e:
                        logger.error(f"{project_name}仓库更新失败, error {e}")
                        raise ValueError(f"{project_name}仓库更新失败")

                if not again:
                    if pull:
                        # 获取拉取前后的差异
                        diff = old_commit.diff(new_commit)
                        # 找出被修改或新增的文件
                        updeted_files = [item.b_path for item in diff if item.change_type in ['M', 'A']]
                    else:
                        updeted_files = copy.deepcopy(self.community_projects_infos[project_name]["unsuccessful_nodes"])
                    for file_path in updeted_files:
                        if os.path.commonpath([file_path, local_resources_dir]) == local_resources_dir:
                            try:
                                file_name = os.path.basename(file_path)
                                node_subject = os.path.splitext(file_name)[0]
                                rsp = self.llm_generator.messages_generate_llm(node_subject, file_path)
                                rsp_json = parse_json(rsp)

                                json_path = os.path.join(local_jsons_dir, f"{file_name}.json")
                                if os.path.exist(json_path):
                                    old_json = load4json(json_path, [])
                                    old_json.extend(rsp_json)
                                else:
                                    old_json = rsp_json
                                save2json(old_json, json_path)
                                self.community_projects_infos[project_name]["successful_nodes"].append(file_path)
                                self.community_projects_infos[project_name]["unsuccessful_nodes"].remove(file_path)
                            except Exception as e:
                                logger.error(f"{file_path}, error {e}")
                                if file_path not in self.community_projects_infos[project_name]["unsuccessful_nodes"]:
                                    self.community_projects_infos[project_name]["unsuccessful_nodes"].append(file_path)
                else:
                    if pull and len(os.listdir(local_jsons_dir)) > 0:
                        raise ValueError(f"重头生成一整个版本时，{version}已存在")
                    
                    if project_name == "ComfyUI-docs":
                        self.comfyUI_docs_again(local_resources_dir, local_jsons_dir)
                    if project_name == "SaltAI-Web-Docs":
                        self.saltAI_web_docs_again(local_resources_dir, local_jsons_dir)
                    if project_name == "comfyui-nodes-docs":
                        self.comfyui_nodes_docs_again(local_resources_dir, local_jsons_dir)
                    if project_name == "comflowy":
                        self.comflowy_again(local_resources_dir, local_jsons_dir)
                
                self.local_community_project_infos[project_name]["json_last_update"] = current_time
        finally:
            self.save_infos()

    def comfyUI_docs_again(self, local_resources_dir: str, local_jsons_dir: str) -> None:
        for item in os.listdir(local_resources_dir):
            temp_path = os.path.join(local_resources_dir, item)
            if item != "media" and os.path.isdir(temp_path) and len(os.listdir(temp_path)) > 0:
                try:
                    for item2 in os.listdir(temp_path):
                        if item2 == "index.md" or item2 == "media":
                            continue
                        elif item2.endswith('.MD') or item2.endswith('.MDX') or item2.endswith('.md') or item2.endswith('.mdx'):
                            md_path = os.path.join(temp_path, item2)
                            if md_path not in self.community_projects_infos["ComfyUI-docs"]["successful_nodes"]:
                                md_name = os.path.splitext(item2)[0]
                                rsp = self.llm_generator.messages_generate_llm(md_name, md_path)
                                rsp_json = parse_json(rsp)

                                json_path = os.path.join(local_jsons_dir, f"{md_name}.json")
                                if os.path.exists(json_path):
                                    old_json = load4json(json_path, [])
                                    old_json.extend(rsp_json)
                                else:
                                    old_json = rsp_json
                                save2json(old_json, json_path)
                                self.community_projects_infos["ComfyUI-docs"]["successful_nodes"].append(md_path)
                                if md_path in self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"]:
                                    self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"].remove(md_path)
                        elif os.path.exists(os.path.join(temp_path, item2)) and len(os.path.join(temp_path, item2)) > 0:
                            temp_path2 = os.path.join(temp_path, item2)
                            for item3 in os.listdir(temp_path2):
                                if item3 == "index.md" or item3 == "media":
                                    continue
                                elif item3.endswith('.MD') or item3.endswith('.MDX') or item3.endswith('.md') or item3.endswith('.mdx'):
                                    md_path = os.path.join(temp_path2, item3)
                                    if md_path not in self.community_projects_infos["ComfyUI-docs"]["successful_nodes"]:
                                        md_name = os.path.splitext(item3)[0]
                                        rsp = self.llm_generator.messages_generate_llm(md_name, md_path)
                                        rsp_json = parse_json(rsp)
                                        
                                        json_path = os.path.join(local_jsons_dir, f"{md_name}.json")
                                        if os.path.exists(json_path):
                                            old_json = load4json(json_path, [])
                                            old_json.extend(rsp_json)
                                        else:
                                            old_json = rsp_json
                                        save2json(old_json, json_path)
                                        self.community_projects_infos["ComfyUI-docs"]["successful_nodes"].append(md_path)
                                        if md_path in self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"]:
                                            self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"].remove(md_path)
                except Exception as e:
                    logger.error(f"{md_path}, error {e}")
                    if md_path not in self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"]:
                        self.community_projects_infos["ComfyUI-docs"]["unsuccessful_nodes"].append(md_path)

    def saltAI_web_docs_again(self, local_resources_dir: str, local_jsons_dir: str) -> None:
        for node in os.listdir(local_resources_dir):
            node_path = os.path.join(local_resources_dir, node)
            if os.path.isdir(node_path) and len(os.listdir(node_path)) > 0:
                for item in os.listdir(node_path):
                    if item == "Nodes":
                        sub_node_dir = os.path.join(node_path, item)
                        for sub_node in os.listdir(sub_node_dir):
                            sub_node_path = os.path.join(sub_node_dir, sub_node)
                            if sub_node_path not in self.community_projects_infos["SaltAI-Web-Docs"]["successful_nodes"]:
                                try: 
                                    sub_node_name = os.path.splitext(sub_node)[0]
                                    rsp = self.llm_generator.messages_generate_llm(sub_node_name, sub_node_path,
                                                                                   system_prompt=system_prompt2, template=template2)
                                    rsp_json = parse_json(rsp)

                                    json_path = os.path.join(local_jsons_dir, f"{node}+{sub_node_name}.json")
                                    if os.path.exists(json_path):
                                        old_json = load4json(json_path, [])
                                        old_json.extend(rsp_json)
                                    else:
                                        old_json = rsp_json
                                    save2json(old_json, json_path)
                                    self.community_projects_infos["SaltAI-Web-Docs"]["successful_nodes"].append(sub_node_path)
                                    if sub_node_path in self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"]:
                                        self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"].remove(sub_node_path)
                                    # break
                                except Exception as e:
                                    logger.error(f'Failed to extract data from file: {sub_node_path}, error: {e}')
                                    if sub_node_path not in self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"]:
                                        self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"].append(sub_node_path)
                    elif item == "index.md":
                        try: 
                            index_path = os.path.join(node_path, item)
                            if index_path not in self.community_projects_infos["SaltAI-Web-Docs"]["successful_nodes"]:
                                rsp = self.llm_generator.messages_generate_llm(sub_node_name, index_path,
                                                                               system_prompt=system_prompt2, template=template2)
                                rsp_json = parse_json(rsp)

                                json_path = os.path.join(local_jsons_dir, f"{node}.json")
                                if os.path.exists(json_path):
                                    old_json = load4json(json_path, [])
                                    old_json.extend(rsp_json)
                                else:
                                    old_json = rsp_json
                                save2json(old_json, json_path)
                                self.community_projects_infos["SaltAI-Web-Docs"]["successful_nodes"].append(index_path)
                                if index_path in self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"]:
                                    self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"].remove(index_path)
                        except Exception as e:
                            logger.error(f'Failed to extract data from file: {index_path}, error: {e}')
                            if index_path not in  self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"]:
                                 self.community_projects_infos["SaltAI-Web-Docs"]["unsuccessful_nodes"].append(index_path)
                    else:
                        continue

    def comfyui_nodes_docs_again(self, local_resources_dir: str, local_jsons_dir: str) -> None:
        for item in os.listdir(local_resources_dir):
            md_path = os.path.join(local_resources_dir, item)
            if not os.path.isdir(md_path):
                name, ext = extract_name_extension(item)
                if ext in ['.MD', '.MDX', '.md', '.mdx'] and md_path not in self.community_projects_infos["comfyui-nodes-docs"]["successful_nodes"]:
                    try:
                        rsp = self.llm_generator.messages_generate_llm(name, md_path,
                                                                       system_prompt=system_prompt_zh,
                                                                       template=template_zh)
                        rsp_json = parse_json(rsp)

                        json_path = os.path.join(local_jsons_dir, f"{name}.json")
                        if os.path.exists(json_path):
                            old_json = load4json(json_path, [])
                            old_json.extend(rsp_json)
                        else:
                            old_json = rsp_json
                        save2json(old_json, json_path)
                        self.community_projects_infos["comfyui-nodes-docs"]["successful_nodes"].append(md_path)
                        if md_path in self.community_projects_infos["comfyui-nodes-docs"]["unsuccessful_nodes"]:
                            self.community_projects_infos["comfyui-nodes-docs"]["unsuccessful_nodes"].remove(md_path)
                        # break
                    except Exception as e:
                        logger.error(f'Failed to extract data from file: {md_path}, error: {e}')
                        if md_path not in self.community_projects_infos["comfyui-nodes-docs"]["unsuccessful_nodes"]:
                            self.community_projects_infos["comfyui-nodes-docs"]["unsuccessful_nodes"].append(md_path)

    def comflowy_again(self, local_resources_dir: str, local_jsons_dir: str) -> None:
        for item in os.listdir(local_resources_dir):
            md_dir = os.path.join(local_resources_dir, item)
            if os.path.isdir(md_dir):
                for md in os.listdir(md_dir):
                    name, ext = extract_name_extension(md)
                    md_path = os.path.join(md_dir, md)
                    if ext in ['.MD', '.MDX', '.md', '.mdx'] and md_path not in self.community_projects_infos["comflowy"]["successful_nodes"]:
                        try:
                            _, flag = extract_name_extension(name)
                            if flag == ".en-US":
                                system_prompt = system_prompt2_index
                                template = template2_index
                            elif flag == ".zh-CN":
                                system_prompt = system_prompt_zh
                                template = template_zh
                            else:
                                raise ValueError("文件标识错误")
                            rsp = self.llm_generator.messages_generate_llm(name, md_path,
                                                                            system_prompt=system_prompt,
                                                                            template=template)
                            rsp_json = parse_json(rsp)

                            json_path = os.path.join(local_jsons_dir, f"{name}.json")
                            if os.path.exists(json_path):
                                old_json = load4json(json_path, [])
                                old_json.extend(rsp_json)
                            else:
                                old_json = rsp_json
                            save2json(old_json, json_path)
                            self.community_projects_infos["comflowy"]["successful_nodes"].append(md_path)
                            if md_path in self.community_projects_infos["comflowy"]["unsuccessful_nodes"]:
                                self.community_projects_infos["comflowy"]["unsuccessful_nodes"].remove(md_path)
                        except Exception as e:
                            logger.error(f'Failed to extract data from file: {md_path}, error: {e}')
                            if md_path not in self.community_projects_infos["comflowy"]["unsuccessful_nodes"]:
                                self.community_projects_infos["comflowy"]["unsuccessful_nodes"].append(md_path)

    def save_infos(self):
        try:
            save2json(self.local_community_project_infos,
                      self.local_community_project_infos_path)
        except Exception as e:
            print(str(e))
            print(self.local_community_project_infos)
            logger.error(self.local_community_project_infos)
            raise ValueError("各社区项目处理信息保存失败")


# TODO 优化各数据块混合方案


if __name__ == "__main__":
    # generator = LLMApiGenerator()
    # eng_text = "hello world"
    # print(generator.eng2zh_llm(eng_text))

    # 基于comfyui-manager维护的节点列表构建nodes-v2数据
    pipeline = DataCollectAndMessagesGeneratePipelineWithComfyuiManager()
    pipeline.refresh_all_repos()
    # pipeline.refresh_all_mds()
    # pipeline.refresh_all_jsons()