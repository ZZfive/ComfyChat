#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2024/06/23 12:19:20
@Author  :   zzfive 
@Desc    :   None
'''
import os
import sys
server_dir = os.path.dirname(__file__)
parent_dir =  os.path.abspath(os.path.join(server_dir, '..'))
sys.path.insert(0, parent_dir)

import re
import time
import socket
import argparse
import threading
from typing import List, Tuple, Generator

import torch
import pytoml
import gradio as gr
import numpy as np
import whisperx

from llm_infer import HybridLLMServer
from retriever import CacheRetriever
from module_gptsovits import get_tts_wav, default_gpt_path, default_sovits_path, set_gpt_weights, set_sovits_weights
from module_comfyui import Option, Choices, Lora, Upscale, UpscaleWithModel, ControlNet, Postprocess, SD, SC, SVD, Extras, Info, send_to
from prompt_templates import PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE, ANSWER_NO_RAG_TEMPLATE, ANSWER_RAG_TEMPLATE, ANSWER_LLM_TEMPLATE, ANSWER_SUFFIXES

# 参数设置
parser = argparse.ArgumentParser(description='ComfyChat Server.')
parser.add_argument(
        '--config-path',
        default=os.path.join(server_dir, 'config.ini'),
        help=  'LLM Server configuration path. Default value is config.ini'  # noqa E501
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
en_retriever = cache.get(fs_id="en", work_dir=os.path.join(server_dir, config["feature_store"]["work_dir"]["en"]),
                         languega="en", config_path=args.config_path)
zh_retriever = cache.get(fs_id="zh", work_dir=os.path.join(server_dir, config["feature_store"]["work_dir"]["zh"]),
                         languega="zh", reject_index_name="index-gpu", config_path=args.config_path)
# asr实例初始化
asr_model = whisperx.load_model(config["whisperx"]["model"], device, compute_type="float16", language='en',
                                download_root=os.path.join(parent_dir, config["whisperx"]["download_root"]))

# ComfyUI组件初始化
if config['comfyui']['enable'] == 1:
    opt = Option(config_path=args.config_path)

    comfyui_dir = os.path.join(parent_dir, opt.comfyui_dir)
    comfyui_main_file = os.path.join(parent_dir, opt.comfyui_file)
    comfyui_main_port = opt.comfyui_port
    sys.path.append(comfyui_dir)

    def start_comfyui(script_path, port, event):
        # 启动服务的函数，例如使用 subprocess 启动服务
        import subprocess
        # 示例命令，替换为实际命令
        cmd = f'python {script_path} --port {port}'
        process = subprocess.Popen(cmd, shell=True)
        
        # 等待服务启动
        while True:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)

        # 服务启动后设置事件
        event.set()

    service_started_event = threading.Event()
    thread = threading.Thread(target=start_comfyui, args=(comfyui_main_file, comfyui_main_port, service_started_event))
    thread.setDaemon(True)
    thread.start()
    # 等待服务启动完成
    service_started_event.wait()

    choices = Choices(opt)
    lora = Lora(opt, choices)
    upscale = Upscale(choices)
    upscale_model = UpscaleWithModel(choices)
    controlnet = ControlNet(opt, choices)
    postprocessor = Postprocess(upscale, upscale_model)
    sd = SD(opt, choices, lora, controlnet, postprocessor)
    sc = SC(opt, choices, postprocessor)
    svd = SVD(opt, choices, postprocessor)
    extras = Extras(opt, choices, postprocessor)
    info = Info(opt, choices, postprocessor)

    # 用于加载comfyui界面
    html_code = f"""
    <iframe src="http://127.0.0.1:{comfyui_main_port}" width="100%" height="1000px"></iframe>
    """


def generate_answer(prompt: str, history: list, lang: str = 'en', backend: str = 'remote', use_rag: bool = False) -> str:
    # 默认不走RAG
    if not use_rag:
        prompt = PROMPT_TEMPLATE["EN_PROMPT_TEMPALTE" if lang == "en" else "ZH_PROMPT_TEMPALTE"].format(question=prompt)

    if use_rag:  # 设置了走RAG
        chunk, context, references = en_retriever.query(prompt) if lang == "en" else zh_retriever.query(prompt)
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
        

# 清除文本中一些不适合被TTS的内容
def clean_text(answer: str, suffixes: List[str] = ANSWER_SUFFIXES) -> str:
    # 清除末尾的固定说明文本
    for suffix in suffixes:
        answer = answer.replace(suffix, "")

    # 清除url
    url_pattern = r'https?://\S+'
    answer = re.sub(url_pattern, '', answer)
    
    return answer


def audio2text(audio_path: str) -> str:
    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(audio, batch_size=16)
    txts = [res['text'] for res in result['segments']]
    text = ' '.join(txts)
    return text


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


# 开启tts功能，将GPT-SoVITS涉及的两个可调模块打开
def toggle_tts_radio(show: bool) -> Tuple[bool, bool]:
    return gr.update(visible=show), gr.update(visible=show)


# 选择local inference后开启选择推理引擎
def toggle_local_engin(backend: str) -> bool:
    if backend == "local":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    

# 开启comfyui界面
def toggle_comfyui(show: bool) -> Tuple[bool, bool]:
    return (gr.update(visible=not show), gr.update(visible=show))
    

# 使用GPT-SoVITS时随着克隆对象的变化设置对应参数
def update_gpt_sovits(selected_option: str) -> Tuple[str]:
    if selected_option == "派蒙":
        return (os.path.join(parent_dir, config["gptsovits"]["gpt_path"]["paimeng"]),
                os.path.join(parent_dir, config["gptsovits"]["sovits_path"]["paimeng"]),
                os.path.join(parent_dir, config["gptsovits"]["wav"]["paimeng"]),
                config["gptsovits"]["prompt"]["paimeng"],
                config["gptsovits"]["language"]["paimeng"])
    elif selected_option == "罗刹":
        return (os.path.join(parent_dir, config["gptsovits"]["gpt_path"]["luocha"]),
                os.path.join(parent_dir, config["gptsovits"]["sovits_path"]["luocha"]),
                os.path.join(parent_dir, config["gptsovits"]["wav"]["luocha"]),
                config["gptsovits"]["prompt"]["luocha"],
                config["gptsovits"]["language"]["luocha"])
    elif selected_option == "胡桃":
        return (os.path.join(parent_dir, config["gptsovits"]["gpt_path"]["hutao"]),
                os.path.join(parent_dir, config["gptsovits"]["sovits_path"]["hutao"]),
                os.path.join(parent_dir, config["gptsovits"]["wav"]["hutao"]),
                config["gptsovits"]["prompt"]["hutao"],
                config["gptsovits"]["language"]["hutao"])
    elif selected_option == "魈":
        return (os.path.join(parent_dir, config["gptsovits"]["gpt_path"]["xiao"]),
                os.path.join(parent_dir, config["gptsovits"]["sovits_path"]["xiao"]),
                os.path.join(parent_dir, config["gptsovits"]["wav"]["xiao"]),
                config["gptsovits"]["prompt"]["xiao"],
                config["gptsovits"]["language"]["xiao"])


# 定义处理选择事件的回调函数
def chatbot_selected2tts(evt: gr.SelectData, use_tts: bool, text_language: str, cut_punc: str, gpt_path: str, sovits_path: str,
                         ref_wav_path: str, prompt_text: str, prompt_language: str) -> Tuple[int, np.ndarray]:
    '''
    gradio版本不同，evt内部结构貌似不同，导致selected_text结构不同
        4.37.2--selected_text是一个字典{'type': 'text', 'value': 'xxxxx'}
        4.32.2--4.32.2就是所选框中的文本
    '''
    global default_gpt_path, default_sovits_path
    selected_index = evt.index  # 获取用户选择的对话条目索引
    selected_text = evt.value   # 获取用户选择的对话条目文本
    # if use_tts and selected_index[1] == 1 and selected_text['type'] == 'text':  # 可能是gradio版本问题，导致此处取值方式存在不同
    #     text = selected_text['value']
    if use_tts and selected_index[1] == 1:
        text = selected_text
        text = clean_text(text)
        if gpt_path != default_gpt_path and sovits_path != default_sovits_path:
            set_gpt_weights(gpt_path)
            set_sovits_weights(sovits_path)
            default_gpt_path = gpt_path
            default_sovits_path = sovits_path
        results = get_tts_wav(text, text_language, cut_punc, ref_wav_path, prompt_text, prompt_language, return_numpy=True)
        return results
    else:
        with open(os.path.join(parent_dir, "audio/wavs/default.wav"), "rb") as audio_file:
            audio_bytes = audio_file.read()
            data = np.frombuffer(audio_bytes, dtype=np.int16)
        return (32000, data)


with gr.Blocks() as demo:
    with gr.Tab("ComfyChat based on LLM"):
        with gr.Row():
            with gr.Column(scale=1):
                backend = gr.Radio(["local", "remote"], value="remote", label="Inference backend")
                local_backend_engin = gr.Radio(["Transformers", "LMDeploy"], value="Transformers", visible=False,
                                               label="The engine used for local inference")
                lang = gr.Radio(["zh", "en"], value="en", label="Language")
                use_rag = gr.Radio([True, False], value=False, label="Turn on RAG")
                use_tts = gr.Checkbox(label="Use TTS based on GPT-SoVITS", info="Left click on the text generated by LLM to convert it to speech")
                # GPT-SoVITS相关组件
                gpt_sovits_voice = gr.Radio(["派蒙", "罗刹", "胡桃", "魈"], value="派蒙", label="Reference audio for GPT-SoVITS", visible=False)
                cut_punc = gr.Textbox(value=",.;?!、，。？！；：…", label="Delimiters for GPT-SoVITS", visible=False)
                # GPT-SoVITS推理时需要的参数，正常情况下保持不可见
                gpt_path = gr.Textbox(value=default_gpt_path, visible=False)
                sovits_path = gr.Textbox(value=default_sovits_path, visible=False)
                ref_wav_path = gr.Textbox(value=os.path.join(parent_dir, config["gptsovits"]["wav"]["paimeng"]), visible=False)
                prompt_text = gr.Textbox(value=config["gptsovits"]["prompt"]["paimeng"], visible=False)
                prompt_language = gr.Textbox(value=config["gptsovits"]["language"]["paimeng"], visible=False)

            with gr.Column(scale=11):
                chatbot = gr.Chatbot(label="ComfyChat",
                                     avatar_images=(("source/comfychat2.webp", 
                                                     "source/comfychat3.webp")))
                msg = gr.Textbox()
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

                # 语音输入
                in_audio = gr.Audio(sources="microphone", type="filepath", label="Voice input")
                audio2text_buttong = gr.Button("Transcribe audio to text")

                # 设置
                out_audio = gr.Audio(label="Click on the reply text to generate the corresponding audio. If TTS is not enabled, the default audio will be returned.",
                                     type="numpy")
            
            # backend.change(toggle_local_engin, inputs=backend, outputs=local_backend_engin)  # 做成动态的比较耗资源，暂时不开启
            use_tts.change(toggle_tts_radio, inputs=use_tts, outputs=[gpt_sovits_voice, cut_punc])
            gpt_sovits_voice.change(update_gpt_sovits, inputs=gpt_sovits_voice, outputs=[gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language])

            audio2text_buttong.click(audio2text, in_audio, msg)
            submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, inputs=[chatbot, lang, backend, use_rag], outputs=chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)

            # 添加 Chatbot.select 事件监听器
            chatbot.select(chatbot_selected2tts, inputs=[use_tts, lang, cut_punc, gpt_path, sovits_path, ref_wav_path,
                                                         prompt_text, prompt_language], outputs=out_audio)
    if config['comfyui']['enable'] == 1:
        with gr.Tab("ComfyUI GUI"):
            turn_on_comfyui = gr.Checkbox(label="Switch to ComfyUI GUI", info="Switch the default raw image interface to ComfyUI GUI")
            with gr.Group() as fixed_workflow:
                with gr.Blocks():
                    # Initial.initialized = gr.Checkbox(value=False, visible=False)
                    with gr.Tab(label="Stable Diffusion"):
                        sd.blocks(sc.enable, svd.enable)
                    if sc.enable is True:
                        with gr.Tab(label="Stable Cascade"):
                            sc.blocks(svd.enable)
                    if svd.enable is True:
                        with gr.Tab(label="Stable Video Diffusion"):
                            svd.blocks()
                    with gr.Tab(label="Extras"):
                        extras.blocks()
                    with gr.Tab(label="Info"):
                        info.blocks()
                    
                    sd.send_to_sd.click(fn=send_to, inputs=[sd.data, sd.index], outputs=[sd.input_image])
                    if sc.enable is True:
                        sd.send_to_sc.click(fn=send_to, inputs=[sd.data, sd.index], outputs=[sc.input_image])
                    if svd.enable is True:
                        sd.send_to_svd.click(fn=send_to, inputs=[sd.data, sd.index], outputs=[svd.input_image])
                    sd.send_to_extras.click(fn=send_to, inputs=[sd.data, sd.index], outputs=[extras.input_image])
                    sd.send_to_info.click(fn=send_to, inputs=[sd.data, sd.index], outputs=[info.input_image])
                    if sc.enable is True:
                        sc.send_to_sd.click(fn=send_to, inputs=[sc.data, sc.index], outputs=[sd.input_image])
                        sc.send_to_sc.click(fn=send_to, inputs=[sc.data, sc.index], outputs=[sc.input_image])
                        if svd.enable is True:
                            sc.send_to_svd.click(fn=send_to, inputs=[sc.data, sc.index], outputs=[svd.input_image])
                        sc.send_to_extras.click(fn=send_to, inputs=[sc.data, sc.index], outputs=[extras.input_image])
                        sc.send_to_info.click(fn=send_to, inputs=[sc.data, sc.index], outputs=[info.input_image])
            comfyui_gui = gr.HTML(html_code, visible=False)

            turn_on_comfyui.change(toggle_comfyui, inputs=turn_on_comfyui, outputs=[fixed_workflow, comfyui_gui])

demo.queue()
demo.launch(server_name="0.0.0.0")