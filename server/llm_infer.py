#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   llm.py
@Time    :   2024/06/19 22:17:47
@Author  :   zzfive 
@Desc    :   None
'''

# TODO 目标，可切换底层和语言的llm推理接口，最终是用于webui中，接受prompt、context和history，返回结果
'''
1 llm初始化、推理整个pipeline搭建起来
2 backend切换
3 中英提示词模板
'''

from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from openai._types import NOT_GIVEN

import config

PROMPT_TEMPLATE = dict(
    ZH_RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    
    EN_RAG_PROMPT_TEMPALTE="""Answer user questions in context. If you don’t know the answer, say you don’t know. Always answer in English.
        Question: {question}
        Reference context：
        ···
        {context}
        ···
        If the given context does not allow you to answer, please respond that there is no such thing in the database and you don't know it.
        Useful answers:""",
    
)


class LikeOpenLLM(ABC):
    def __init__(self, base_url: str, api_key: str, model: str, **kwargs) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model

    @abstractmethod
    def chat(self, prompt: str, context: str, history: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError
    

class KimiLLM(LikeOpenLLM):
    def __init__(self, base_url: str, api_key: str, model: str, **kwargs) -> None:
        super().__init__(base_url, api_key, model, **kwargs)

    def chat(self, prompt: str, context: str, history: List[Dict[str, str]] = [], is_zh: bool = False, **kwargs) -> str:
        history.append({
            "role": "user",
            "content": PROMPT_TEMPLATE["ZH_RAG_PROMPT_TEMPALTE" if is_zh else "EN_RAG_PROMPT_TEMPALTE"].format(question=prompt, context=context)
        })

        temperature = kwargs.get('temperature', NOT_GIVEN)
        top_p = kwargs.get('top_p', NOT_GIVEN)
        frequency_penalty = kwargs.get('frequency_penalty', NOT_GIVEN)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty
        )
        return response.choices[0].message.content
    

class LocalLLM(ABC):
    def __init__(self, model_path: str, **kwargs) -> None:
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()

    @abstractmethod
    def chat(self, prompt: str, context: str, history: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError
    

class LocalLLM(LocalLLM):
    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        

    def chat(self, prompt: str, context: str, history: List[Dict[str, str]] = [], is_zh: bool = False, **kwargs) -> str:
        prompt = PROMPT_TEMPLATE["ZH_RAG_PROMPT_TEMPALTE" if is_zh else "EN_RAG_PROMPT_TEMPALTE"].format(question=prompt, context=context)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response
    

if __name__ == "__main__":
    llm = KimiLLM(base_url="https://api.moonshot.cn/v1", api_key=config.MOONSHOT_API_KEY, model="moonshot-v1-8k")
    # print(llm.chat("What is the capital of France?", "The capital of France is Paris.", is_zh=True))
    context = '''
    # Diffusers Loader

    ![Diffusers Loader node](media/DiffusersLoader.svg){ align=right width=450 }

    The Diffusers Loader node can be used to load a diffusion model from diffusers.

    ## inputs

    `model_path`

    :   path to the diffusers model.

    ## outputs

    `MODEL`

    :   The model used for denoising latents.

    `CLIP`

    :   The CLIP model used for encoding text prompts.

    `VAE`

    :   The VAE model used for encoding and decoding images to and from latent space.

    ## example

    example usage text with workflow image
    '''
    print(llm.chat("What is the Diffusers Loader node?", context))