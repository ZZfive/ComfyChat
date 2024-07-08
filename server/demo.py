#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2024/06/23 12:19:20
@Author  :   zzfive 
@Desc    :   None
'''
import os
# os.environ["HF_HOME"] = r"D:\cache"
# os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\cache"
# os.environ["TRANSFORMERS_CACHE"] = r"D:\cache"
import sys
sys.path.append('/root/code/ComfyChat')
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
from audio.gptsovits import get_tts_wav, default_gpt_path, default_sovits_path, set_gpt_weights, set_sovits_weights
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
# chattts实例初始化
chattts_model = ChatTTS.Chat()
chattts_model.load(compile=False, source='huggingface')


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


def text2audio_chattts(text: str, seed: int = 42, refine_text_flag: bool = True) -> None:
    torch.manual_seed(seed)# 设置采样音色的随机种子
    rand_spk = chattts_model.sample_random_speaker()  # 从高斯分布中随机采样出一个音色
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,
        temperature=.3,
        top_P=0.7,
        top_K=20)
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',)

    torch.manual_seed(random.randint(1, 100000000))

    if refine_text_flag:
        text = chattts_model.infer(text, 
                                skip_refine_text=False,
                                refine_text_only=True,
                                params_refine_text=params_refine_text,
                                params_infer_code=params_infer_code)

    wavs = chattts_model.infer(text,
                            skip_refine_text=True,
                            params_refine_text=params_refine_text,
                            params_infer_code=params_infer_code)
    audio_data = np.array(wavs[0]).flatten()

    return (24000, audio_data)

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


# 控制tts模型选择模块可见
def toggle_tts_radio(show: bool):
    return gr.update(visible=show)

# 控制tts模型相关组件可见
def toggle_tts_components(selected_option: str):
    if selected_option == "Chattts":
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
    elif selected_option == "GPT-SoVITS":
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True))
    

# 使用GPT-SoVITS时随着克隆对象的变化设置对应参数
def update_gpt_sovits(selected_option: str) -> Tuple[str]:
    if selected_option == "默认":
        return (default_gpt_path, default_sovits_path, "/root/code/ComfyChat/audio/wavs/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav",
                "疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？", "zh")
    elif selected_option == "派蒙":
        return (default_gpt_path, default_sovits_path, "/root/code/ComfyChat/audio/wavs/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav",
                "疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？", "zh")
    elif selected_option == "魈":
        return (default_gpt_path, default_sovits_path, "/root/code/ComfyChat/audio/wavs/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav",
                "疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？", "zh")


# 定义处理选择事件的回调函数
def chatbot_selected2tts(evt: gr.SelectData, use_tts: bool, tts_model: str, chattts_audio_seed: int,
                         text_language: str, cut_punc: str, gpt_path: str, sovits_path: str, ref_wav_path: str,
                         prompt_text: str, prompt_language: str) -> List[Tuple[int, np.ndarray] | str]:
    selected_index = evt.index  # 获取用户选择的对话条目索引
    selected_text = evt.value   # 获取用户选择的对话条目文本
    if use_tts and selected_index[1] == 1 and selected_text['type'] == 'text':
        text = selected_text['value']
        if tts_model == 'chattts':
            results = text2audio_chattts(text, chattts_audio_seed)
            return results
        elif tts_model == "gpt-sovit":
            if gpt_path != default_gpt_path and sovits_path != default_sovits_path:
                set_gpt_weights(gpt_path)
                set_sovits_weights(sovits_path)
            results = get_tts_wav(text, text_language, cut_punc, ref_wav_path, prompt_text, prompt_language, return_numpy=True)
            return results
    else:
        return (None, None)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            backend = gr.Radio(["local", "remote"], value="remote", label="Inference backend")
            lang = gr.Radio(["zh", "en"], value="en", label="Language")
            use_rag = gr.Radio([True, False], value=False, label="Turn on RAG")
            use_tts = gr.Checkbox(label="Use TTS", info="Turn on TTS")
            tts_model = gr.Radio(["Chattts", "GPT-SoVITS"], visible=False)
            # Chattts相关组件
            chattts_audio_seed = gr.Slider(min=1, max=100000000, step=1, value=42, label="Audio seed for Chattts", visible=False)
            # GPT-SoVITS相关组件
            gpt_sovits_voice = gr.Radio(["默认", "派蒙", "魈"], value="默认", label="Reference audio for GPT-SoVITS", visible=False)
            cut_punc = gr.Textbox(value=",.;?!、，。？！；：…", label="Delimiters for GPT-SoVITS", interactive=False)
            # GPT-SoVITS推理时需要的参数，正常情况下保持不可见
            gpt_path = gr.Textbox(value=default_gpt_path, interactive=False)
            sovits_path = gr.Textbox(value=default_sovits_path, interactive=False)
            ref_wav_path = gr.Textbox(value="/root/code/ComfyChat/audio/wavs/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav", interactive=False)
            prompt_text = gr.Textbox(value="疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？", interactive=False)
            prompt_language = gr.Textbox(value="zh", interactive=False)

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
            
        use_tts.change(toggle_tts_radio, inputs=use_tts, outputs=tts_model)
        tts_model.change(toggle_tts_components, inputs=tts_model, outputs=[chattts_audio_seed, gpt_sovits_voice, cut_punc])
        gpt_sovits_voice.change(update_gpt_sovits, inputs=gpt_sovits_voice, outputs=[gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language])

        audio2text_buttong.click(audio2text, in_audio, msg)
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, inputs=[chatbot, lang], outputs=chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

        # 添加 Chatbot.select 事件监听器
        chatbot.select(chatbot_selected2tts, inputs=[use_tts, tts_model, chattts_audio_seed, lang, cut_punc, gpt_path,
                                                     sovits_path, ref_wav_path, prompt_text, prompt_language], outputs=out_audio)

demo.queue()
demo.launch(server_name="0.0.0.0")