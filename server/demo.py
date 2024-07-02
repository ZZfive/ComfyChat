#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2024/06/23 12:19:20
@Author  :   zzfive 
@Desc    :   None
'''
import os
os.environ["HF_HOME"] = r"D:\cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\cache"
import time
import random
import argparse
from typing import List, Tuple, Generator, Dict

import torch
import torchaudio
import pytoml
import gradio as gr
import numpy as np
from whispercpp import Whisper
import whisperx

from llm_infer import HybridLLMServer
from retriever import CacheRetriever
from audio.ChatTTS import ChatTTS
from prompt_templates import PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE, ANSWER_NO_RAG_TEMPLATE, ANSWER_RAG_TEMPLATE, ANSWER_LLM_TEMPLATE

# 参数设置
parser = argparse.ArgumentParser(description='ComfyChat Server.')
parser.add_argument(
        '--config-path',
        default='/root/code/ComfyChat/server/config.ini',
        help=  'LLM Server configuration path. Default value is config.ini'  # noqa E501
    )
parser.add_argument(
        '--asr-model',
        default='whispercpp'
    )
parser.add_argument(
        '--tts-model',
        default='chattts'
    )
args = parser.parse_args()

# 加载配置
with open(args.config_path, encoding='utf-8') as f:
    config = pytoml.load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
# 推理llm实例
llm_config = config['llm']
llm = HybridLLMServer(llm_config)
# RAG检索实例
cache = CacheRetriever(config_path=args.config_path)
retriever = cache.get(config_path=args.config_path)
# asr实例初始化
if args.asr_model == "whispercpp":
    asr_model = Whisper.from_pretrained("base")
elif args.asr_model == "whisperx":
    asr_model = whisperx.load_model("large-v2", device, compute_type="float16", language='en',
                            download_root='/root/code/ComfyChat/weights')
else:
    raise ValueError(f"{args.asr_model} is not supported")
# tts实例初始化
if args.tts_model == "chattts":
    tts_model = ChatTTS.Chat()
    tts_model.load(compile=False, source='huggingface')
elif args.asr_model == "whisperx":
    tts_model = None
else:
    raise ValueError(f"{args.tts_model} is not supported")


def generate_answer(prompt: str, history: list, lang: str = 'en', backend: str = 'remote', use_rag: bool = False) -> str:
    # 默认不走RAG
    prompt = PROMPT_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(question=prompt)

    if use_rag:  # 设置了走RAG
        chunk, context, references = retriever.query(prompt)
        if chunk is not None:  # 当RAG检索到相关上下文
            prompt = RAG_PROMPT_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(question=prompt, context=chunk)

    ans, err = llm.generate_response(prompt, history, backend)

    if err:
        return err
    else:
        if use_rag:
            if chunk is None:
                return ANSWER_NO_RAG_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(answer=ans)
            else:
                return ANSWER_RAG_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(answer=ans)
        else:
            return ANSWER_LLM_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(answer=ans)


def audio2text(audio_path: str) -> str:
    audio = whisperx.load_audio(audio_path)
    if args.asr_model == "whispercpp":
        text = asr_model.transcribe(audio)
    elif args.asr_model == "whisperx":
        result = asr_model.transcribe(audio, batch_size=16)
        # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        txts = [res['text'] for res in result['segments']]
        text = ' '.join(txts)
    else:
        raise ValueError(f"{args.asr_model} is not supported")
    return text


def text2audio(text: str, seed: int = 42, refine_text_flag: bool = True) -> None:
    torch.manual_seed(seed)# 设置采样音色的随机种子
    if args.tts_model == "chattts":
        rand_spk = tts_model.sample_random_speaker()  # 从高斯分布中随机采样出一个音色
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            temperature=.3,
            top_P=0.7,
            top_K=20)
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',)

        torch.manual_seed(random.randint(1, 100000000))

        if refine_text_flag:
            text = tts_model.infer(text, 
                                   skip_refine_text=False,
                                   refine_text_only=True,
                                   params_refine_text=params_refine_text,
                                   params_infer_code=params_infer_code)

        wavs = tts_model.infer(text,
                               skip_refine_text=True,
                               params_refine_text=params_refine_text,
                               params_infer_code=params_infer_code)
        audio_data = np.array(wavs[0]).flatten()
    elif args.tts_model == "sovits":
        pass
    else:
        raise ValueError(f"{args.tts_model} is not supported")
    
    text_data = text[0] if isinstance(text, list) else text
    return [(24000, audio_data), text_data]

def user(user_message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        return "", history + [(user_message, None)]


def bot(chatbot_history: List[Tuple[str, str]], lang: str = 'en', backend: str = 'remote', use_rag: bool = False) -> Generator[str, None, None]:
    history = chatbot_history[:-1]
    prompt = chatbot_history[-1][0]
    bot_message = generate_answer(prompt, history, lang, backend, use_rag)
    # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])  # 此处替换为generate_answer
    chatbot_history[-1][1] = ""
    for character in bot_message:
        chatbot_history[-1][1] += character
        time.sleep(0.05)
        yield chatbot_history


# 定义处理选择事件的回调函数
def chatbot_selected2tts(evt: gr.SelectData, use_tts: bool, audio_seed: int) -> List[Tuple[int, np.ndarray] | str]:
    selected_index = evt.index  # 获取用户选择的对话条目索引
    selected_text = evt.value   # 获取用户选择的对话条目文本
    if use_tts and selected_index[1] == 1 and selected_text['type'] == 'text':
        text = selected_text['value']
        results = text2audio(text, audio_seed)
        return results
    else:
        return [(None, None), None]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            backend = gr.Radio(["local", "remote"], value="remote", label="Inference backend")
            lang = gr.Radio(["zh", "en"], value="en", label="Language")
            use_rag = gr.Radio([True, False], value=False, label="Turn on RAG")
            use_tts = gr.Radio([True, False], value=False, label="Turn on TTS")
            audio_seed = gr.Slider(min=1, max=100000000, step=1, value=42, label="Audio seed")
        with gr.Column(scale=11):
            chatbot = gr.Chatbot(label="ComfyChat")
            msg = gr.Textbox(interactive=False)
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

            # 语音输入
            in_audio = gr.Audio(sources="microphone", type="filepath")
            audio2text_buttong = gr.Button("audio transcribe to text")

            # 设置
            out_audio = gr.Audio(label="Click on the reply text to generate the corresponding audio",
                                 type="numpy")
            out_audio_text = gr.Textbox(visible=False)

        audio2text_buttong.click(audio2text, in_audio, msg)
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

        # 添加 Chatbot.select 事件监听器
        chatbot.select(chatbot_selected2tts, inputs=[use_tts, audio_seed], outputs=[out_audio, out_audio_text])

demo.queue()
demo.launch(server_name="0.0.0.0")