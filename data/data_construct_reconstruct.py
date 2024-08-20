#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2024/08/19 23:45:56
@Author  :   zzfive 
@Desc    :   None
'''
import time
from datetime import datetime, timedelta

import requests
from openai import OpenAI

import config
from prompt_templates import system_prompt1, template1
from utils import create_logger


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
                          "defualt_model": "google/gemma-7b-it:free",
                          "api_key": config.OPENROUTER_API_KEY},
            "siliconflow":{"base_url": "https://api.siliconflow.cn/v1/chat/completions",
                           "defualt_model": "deepseek-ai/deepseek-v2-chat",
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
    

# TODO 基于comfyui-manager构建messages的pipeline
class MessagesGeneratePipelineWithComfyuiManager:
    pass

# TODO 简化当前对四个开源社区的数据提炼过程


# TODO 优化各数据块混合方案


if __name__ == "__main__":
    generator = LLMApiGenerator()
    eng_text = "hello world"
    print(generator.eng2zh_llm(eng_text))