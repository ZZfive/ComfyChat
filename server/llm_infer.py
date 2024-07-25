#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   llm.py
@Time    :   2024/06/19 22:17:47
@Author  :   zzfive 
@Desc    :   None
'''

import time
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pytoml
import requests
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

from util import create_logger

logger = create_logger("llm_infer")


# 构建正常的请求messages
def build_messages(prompt: str, history: List[Tuple[str, str]], system: str = None) -> List[Dict[str, str]]:
    messages = []
    if system is not None and len(system) > 0:
        messages.append({'role': 'system', 'content': system})
    for item in history:
        messages.append({'role': 'user', 'content': item[0]})
        messages.append({'role': 'assistant', 'content': item[1]})
    messages.append({'role': 'user', 'content': prompt})
    return messages


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


# 本地模型推理的一个包装类
class LocalTransformersInferenceWrapper:
    """A class to wrapper kinds of local LLM framework with Transformers."""

    def __init__(self, model_path: str) -> None:
        """Init model handler."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)

        if 'qwen2' in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype='auto',
                device_map='auto',
                trust_remote_code=True).eval()
        elif 'qwen1.5' in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='auto', trust_remote_code=True).eval()
        elif 'qwen' in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto',
                trust_remote_code=True,
                use_cache_quantization=True,
                use_cache_kernel=True,
                use_flash_attn=False).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype='auto').eval()

    def chat(self, prompt: str, history: List[Tuple[str, str]] = []) -> str:
        """Generate a response from local LLM.

        Args:
            prompt (str): The prompt for inference.
            history (list): List of previous interactions.

        Returns:
            str: Generated response.
        """
        output_text = ''

        if type(self.model).__name__ == 'Qwen2ForCausalLM':
            messages = build_messages(
                prompt=prompt,
                history=history,
                system='You are a helpful assistant')  # noqa E501
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text],
                                          return_tensors='pt').to('cuda')
            generated_ids = self.model.generate(model_inputs.input_ids,
                                                max_new_tokens=512,
                                                top_k=1)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids)
            ]

            output_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)[0]
        else:
            if '请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。' in prompt:
                prompt = '你是一个语言专家，擅长分析语句并打分。\n' + prompt
                output_desc, _ = self.model.chat(self.tokenizer, prompt, history, top_k=1, do_sample=False)
                prompt = '"{}"\n请仔细阅读上面的内容，最后的得分是多少？'.format(output_desc)
                output_text, _ = self.model.chat(self.tokenizer, prompt, history)
            else:
                output_text, _ = self.model.chat(self.tokenizer,
                                                 prompt,
                                                 history,
                                                 top_k=1,
                                                 do_sample=False)
        return output_text
    

class LocalLmdeployInferenceWrapper:
    """A class to wrapper kinds of local LLM framework with LMDeploy."""
    
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.backend_config = TurbomindEngineConfig(cache_max_entry_count=0.3,
                                                    session_len=8192)
        self.pipeline = pipeline(self.model_path, backend_config=self.backend_config,
                                 chat_template_config=ChatTemplateConfig(model_name="internlm2")  # 支持的对话模型通过lmdeploy list查看 TODO 需要针对不同模型设置对应模板，不然会报错
                                 )

    def chat(self, prompt: str, history: List[Tuple[str, str]] = []) -> str:
        gen_config = GenerationConfig(top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=1024)
        messages = build_messages(prompt=prompt,
                                  history=history,
                                  system='You are a helpful assistant')
        response = self.pipeline(messages,  gen_config=gen_config)
        
        return response.text
    

class HybridLLMServer:
    """A class to handle server-side interactions with a hybrid language learning model (LLM) service.

    This class is responsible for initializing the local and remote LLMs,
    generating responses from these models as per the provided configuration,
    and handling retries in case of failures.
    """

    def __init__(self,
                 llm_config: Dict,
                 device: str = 'cuda',
                 retry: int = 2) -> None:
        """Initialize the HybridLLMServer with the given configuration, device, and number of retries."""
        self.device = device
        self.retry = retry
        self.llm_config = llm_config
        self.server_config = llm_config['server']
        self.enable_remote = llm_config['enable_remote']
        self.enable_local = llm_config['enable_local']

        self.local_max_length = self.server_config['local_llm_max_text_length']
        self.remote_max_length = self.server_config['remote_llm_max_text_length']
        self.remote_type = self.server_config['remote_type']

        model_path = self.server_config['local_llm_path']

        _rpm = 500
        if 'rpm' in self.server_config:
            _rpm = self.server_config['rpm']
        self.rpm = RPM(_rpm)
        self.token = ('', 0)

        if self.enable_local:
            self.inference = LocalTransformersInferenceWrapper(model_path)
        else:
            self.inference = None
            logger.warning('local LLM disabled.')

    def call_internlm(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        """See https://internlm.intern-ai.org.cn/api/document for internlm remote api."""
        url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions'

        now = time.time()

        header = {
            'Content-Type': 'application/json',
            'Authorization': self.server_config['remote_api_key']
        }

        logger.info('prompt length {}'.format(len(prompt)))

        messages = []
        for item in history:
            messages.append({'role': 'user', 'text': item[0]})
            messages.append({'role': 'assistant', 'text': item[1]})
        messages.append({'role': 'user', 'text': prompt})

        data = {
            'model': 'internlm2-latest',
            'messages': messages,
            'n': 1,
            'disable_report': False,
            'top_p': 0.9,
            'temperature': 0.8,
            'request_output_len': 2048
        }

        output_text = ''
        self.rpm.wait()

        res_json = requests.post(url,
                                 headers=header,
                                 data=json.dumps(data),
                                 timeout=120).json()
        logger.debug(res_json)
        if 'msgCode' in res_json:
            if res_json['msgCode'] == 'A0202':
                logger.error(
                    'Token error, check it starts with "Bearer " or not ?')
                return ''

        res_data = res_json['choices'][0]['message']['content']
        logger.debug(res_json['choices'])
        if len(res_data) < 1:
            logger.error('debug:')
            logger.error(res_json)
            return output_text
        output_text = res_data

        logger.info(output_text)
        if '仩嗨亾笁潪能實験厔' in output_text:
            raise Exception('internlm model waterprint !!!')
        return output_text

    def call_kimi(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        """Generate a response from Kimi (a remote LLM).

        Args:
            prompt (str): The prompt to send to Kimi.
            history (list): List of previous interactions.

        Returns:
            str: Generated response from Kimi.
        """
        self.rpm.wait()

        client = OpenAI(
            api_key=self.server_config['remote_api_key'],
            base_url='https://api.moonshot.cn/v1',
        )

        SYSTEM = '你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一些涉及恐怖主义，种族歧视，黄色暴力，政治宗教等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。'  # noqa E501
        # 20240531 hacking for kimi API incompatible
        # it is very very tricky, please do not change this magic prompt !!!
        if '请仔细阅读以上内容，判断句子是否是个有主题的疑问句' in prompt:
            SYSTEM = '你是一个语文专家，擅长对句子的结构进行分析'

        messages = build_messages(prompt=prompt,
                                  history=history,
                                  system=SYSTEM)

        logger.debug('remote api sending: {}'.format(messages))
        model = self.server_config['remote_llm_model']

        if model == 'auto':
            prompt_len = len(prompt)
            if prompt_len <= int(8192 * 1.5) - 1024:
                model = 'moonshot-v1-8k'
            elif prompt_len <= int(32768 * 1.5) - 1024:
                model = 'moonshot-v1-32k'
            else:
                prompt = prompt[0:int(128000 * 1.5) - 1024]
                model = 'moonshot-v1-128k'

        logger.info('choose kimi model {}'.format(model))

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content

    def call_step(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        """Generate a response from step, see https://platform.stepfun.com/docs/overview/quickstart.

        Args:
            prompt (str): The prompt to send to LLM.
            history (list): List of previous interactions.

        Returns:
            str: Generated response from LLM.
        """
        client = OpenAI(
            api_key=self.server_config['remote_api_key'],
            base_url='https://api.stepfun.com/v1',
        )

        SYSTEM = '你是由阶跃星辰提供的AI聊天助手，你擅长中文，英文，以及多种其他语言的对话。在保证用户数据安全的前提下，你能对用户的问题和请求，作出快速和精准的回答。同时，你的回答和建议应该拒绝黄赌毒，暴力恐怖主义的内容'  # noqa E501
        messages = build_messages(prompt=prompt,
                                  history=history,
                                  system=SYSTEM)

        logger.debug('remote api sending: {}'.format(messages))

        model = self.server_config['remote_llm_model']

        if model == 'auto':
            prompt_len = len(prompt)
            if prompt_len <= int(8192 * 1.5) - 1024:
                model = 'step-1-8k'
            elif prompt_len <= int(32768 * 1.5) - 1024:
                model = 'step-1-32k'
            elif prompt_len <= int(128000 * 1.5) - 1024:
                model = 'step-1-128k'
            else:
                prompt = prompt[0:int(256000 * 1.5) - 1024]
                model = 'step-1-256k'

        logger.info('choose step model {}'.format(model))

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content

    def call_gpt(self,
                 prompt: str,
                 history: List[Tuple[str, str]],
                 base_url: str = None,
                 system: str = None) -> str:
        """Generate a response from openai API.

        Args:
            prompt (str): The prompt to send to openai API.
            history (list): List of previous interactions.

        Returns:
            str: Generated response from RPC.
        """
        if base_url is not None:
            client = OpenAI(api_key=self.server_config['remote_api_key'],
                            base_url=base_url)
        else:
            client = OpenAI(api_key=self.server_config['remote_api_key'])

        messages = build_messages(prompt=prompt,
                                  history=history,
                                  system=system)

        logger.debug('remote api sending: {}'.format(messages))
        completion = client.chat.completions.create(
            model=self.server_config['remote_llm_model'],
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content

    def call_deepseek(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        """Generate a response from deepseek (a remote LLM).

        Args:
            prompt (str): The prompt to send.
            history (list): List of previous interactions.

        Returns:
            str: Generated response.
        """
        client = OpenAI(
            api_key=self.server_config['remote_api_key'],
            base_url='https://api.deepseek.com/v1',
        )

        messages = build_messages(
            prompt=prompt,
            history=history,
            system='You are a helpful assistant')  # noqa E501

        logger.debug('remote api sending: {}'.format(messages))
        completion = client.chat.completions.create(
            model=self.server_config['remote_llm_model'],
            messages=messages,
            temperature=0.1,
        )
        return completion.choices[0].message.content

    def call_zhipuai(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        """Generate a response from zhipuai (a remote LLM).

        Args:
            prompt (str): The prompt to send.
            history (list): List of previous interactions.

        Returns:
            str: Generated response.
        """
        client = OpenAI(
            api_key=self.server_config['remote_api_key'],
            base_url='https://open.bigmodel.cn/api/paas/v4/',
        )

        messages = build_messages(prompt=prompt, history=history)  # noqa E501

        logger.debug('remote api sending: {}'.format(messages))
        completion = client.chat.completions.create(
            model=self.server_config['remote_llm_model'],
            messages=messages,
            temperature=0.1,
        )
        return completion.choices[0].message.content

    def call_siliconcloud(self, prompt: str, history: List[Tuple[str, str]]) -> str:
        self.rpm.wait()

        url = 'https://api.siliconflow.cn/v1/chat/completions'

        token = self.server_config['remote_api_key']
        if not token.startswith('Bearer '):
            token = 'Bearer ' + token
        headers = {
            'content-type': 'application/json',
            'accept': 'application/json',
            'authorization': token
        }

        messages = build_messages(prompt=prompt, history=history)

        payload = {
            'model': self.server_config['remote_llm_model'],
            'stream': False,
            'messages': messages,
            'temperature': 0.1
        }
        response = requests.post(url, json=payload, headers=headers)
        logger.debug(response.text)
        resp_json = response.json()
        text = resp_json['choices'][0]['message']['content']
        return text

    def generate_response(self, prompt: str, history: List[Tuple[str, str]] = [], backend='local') -> Tuple[str, str]:
        """Generate a response from the appropriate LLM based on the configuration. If failed, use exponential backoff.

        Args:
            prompt (str): The prompt to send to the LLM.
            history (list, optional): List of previous interactions. Defaults to [].  # noqa E501
            remote (bool, optional): Flag to determine whether to use a remote server. Defaults to False.  # noqa E501
            backend (str): LLM type to call. Support 'local', 'remote' and specified LLM name ('kimi', 'deepseek' and so on)

        Returns:
            str: Generated response from the LLM.
        """
        output_text = ''
        error = ''
        time_tokenizer = time.time()

        if backend == 'local' and self.inference is None:
            logger.error(
                "!!! fatal error.  !!! \n Detect `enable_local=0` in `config.ini` while backend='local', please immediately stop the service and check it. \n For this request, autofix the backend to '{}' and proceed."
                .format(self.server_config['remote_type']))
            backend = self.server_config['remote_type']

        if backend == 'remote':
            # not specify remote LLM type, use config
            backend = self.server_config['remote_type']

        if backend == 'local':
            prompt = prompt[:self.local_max_length]
            """# Caution: For the results of this software to be reliable and verifiable,  # noqa E501
            it's essential to ensure reproducibility. Thus `GenerationMode.GREEDY_SEARCH`  # noqa E501
            must enabled."""

            output_text = self.inference.chat(prompt, history)

        else:
            prompt = prompt[:self.remote_max_length]

            life = 0
            while life < self.retry:  # 重试

                try:
                    if backend == 'kimi':
                        output_text = self.call_kimi(prompt=prompt,
                                                     history=history)
                    elif backend == 'deepseek':
                        output_text = self.call_deepseek(prompt=prompt,
                                                         history=history)
                    elif backend == 'zhipuai':
                        output_text = self.call_zhipuai(prompt=prompt,
                                                        history=history)
                    elif backend == 'step':
                        output_text = self.call_step(prompt=prompt,
                                                     history=history)
                    elif backend == 'xi-api' or backend == 'gpt':
                        base_url = None
                        system = None
                        if backend == 'xi-api':
                            base_url = 'https://api.xi-ai.cn/v1'
                            system = 'You are a helpful assistant.'
                        output_text = self.call_gpt(prompt=prompt,
                                                    history=history,
                                                    base_url=base_url,
                                                    system=system)
                    elif backend == 'internlm':
                        output_text = self.call_internlm(prompt=prompt,
                                                         history=history)
                    elif backend == 'siliconcloud':
                        output_text = self.call_siliconcloud(prompt=prompt,
                                                             history=history)
                    else:
                        error = 'unknown backend {}'.format(backend)
                        logger.error(error)

                    # skip retry
                    break

                except Exception as e:
                    # exponential backoff
                    error = str(e)
                    logger.error(error)

                    if 'Error code: 401' in error or 'invalid api_key' in error:
                        break  # 此类两种错误就不重试，直接退出

                    life += 1
                    randval = random.randint(1, int(pow(2, life)))
                    time.sleep(randval)

        # logger.debug((prompt, output_text))
        time_finish = time.time()

        logger.debug('Q:{} A:{} \t\t backend {} timecost {} '.format(
            prompt[-100: -1], output_text, backend,
            time_finish - time_tokenizer))
        return output_text, error
    

if __name__ == "__main__":
    # with open("config.ini", encoding='utf-8') as f:
    #     llm_config = pytoml.load(f)['llm']
    # llm = HybridLLMServer(llm_config)
    # # print(llm.chat("What is the capital of France?", "The capital of France is Paris.", is_zh=True))
    # context = '''
    # # Diffusers Loader

    # ![Diffusers Loader node](media/DiffusersLoader.svg){ align=right width=450 }

    # The Diffusers Loader node can be used to load a diffusion model from diffusers.

    # ## inputs

    # `model_path`

    # :   path to the diffusers model.

    # ## outputs

    # `MODEL`

    # :   The model used for denoising latents.

    # `CLIP`

    # :   The CLIP model used for encoding text prompts.

    # `VAE`

    # :   The VAE model used for encoding and decoding images to and from latent space.

    # ## example

    # example usage text with workflow image
    # '''
    # question = "What is the Diffusers Loader node?"
    # prompt = PROMPT_TEMPLATE["EN_RAG_PROMPT_TEMPALTE"].format(question=question, context=context)
    # print(llm.generate_response(prompt))

    # rpm = RPM()
    # tmp = rpm.get_minute_slot()
    # print(type(tmp))
    # print(tmp)

    model_path = "/group_share/finetune/v2_1_8/final_model"
    # inference = LocalTransformersInferenceWrapper(model_path)
    inference = LocalLmdeployInferenceWrapper(model_path)
    res = inference.chat(prompt="你是谁？")
    print(res)
    res = inference.chat(prompt="ComfyUI是什么？")
    print(res)