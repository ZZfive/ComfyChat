#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gptsovits.py
@Time    :   2024/06/30 13:37:43
@Author  :   zzfive 
@Desc    :   None
'''
import os
import re
import subprocess
import soundfile as sf
from io import BytesIO
from time import time as ttime
from typing import Any, Tuple, List, Union, Dict

import sys
now_dir = '/root/code/ComfyChat/audio'
sys.path.append(now_dir)
sys.path.append("%s/GPT-SoVITS" % (now_dir))
sys.path.append("%s/GPT-SoVITS/GPT_SoVITS" % (now_dir))
parent_dir =  os.path.abspath(os.path.join(now_dir, '..'))

import pytoml
import torch
import librosa
import LangSegment
import numpy as np
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer

from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from tools.my_utils import load_audio

dict_language = {
    "中文": "all_zh",
    "英文": "en",
    "日文": "all_ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto",  # 多语种启动切分识别语种
    "all_zh": "all_zh",
    "en": "en",
    "all_ja": "all_ja",
    "zh": "zh",
    "ja": "ja",
    "auto": "auto",
}
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True
stream_mode = "close"
media_type = "wav"
default_gpt_path = "/root/code/ComfyChat/weights/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
default_sovits_path = "/root/code/ComfyChat/weights/GPT_SoVITS/pretrained_models/s2G488k.pth"
bert_path = "/root/code/ComfyChat/weights/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
cnhubert_base_path = "/root/code/ComfyChat/weights/GPT_SoVITS/pretrained_models/chinese-hubert-base"
default_cut_punc = ",.;?!、，。？！；：…"  # 文本切分符号设定, 符号范围
# gpt模型相关全局变量
hz = 50
max_sec = None
t2s_model = None
gpt_config = None
# SoVITS模型相关全局变量
vq_model = None
hps = None

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model()

if is_half:
    bert_model = bert_model.half().to(device)
    ssl_model = ssl_model.half().to(device)
else:
    bert_model = bert_model.to(device)
    ssl_model = ssl_model.to(device)


class DictToAttrRecursive:
    def __init__(self, input_dict: Any) -> None:
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


def set_gpt_weights(gpt_path: str) -> None:
    global max_sec, t2s_model, gpt_config
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    gpt_config = dict_s1["config"]  # dict
    max_sec = gpt_config["data"]["max_sec"]  # int
    t2s_model = Text2SemanticLightningModule(gpt_config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()


def set_sovits_weights(sovits_path: str) -> None:
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    model_params_dict = vars(hps.model)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **model_params_dict
    )
    if ("pretrained" not in sovits_path):  # 推理时不需要enc_q
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    vq_model.load_state_dict(dict_s2["weight"], strict=False)


set_gpt_weights(default_gpt_path)
set_sovits_weights(default_sovits_path)


def get_bert_feature(text: str, word2ph: List[int]) -> torch.tensor:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


def clean_text_inf(text: str, language: str) -> Tuple[List[int] | str]:
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


def get_bert_inf(phones: List[int], word2ph: List[int], norm_text: str, language: str) -> torch.tensor:
    language = language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


def get_phones_and_bert(text: str, language: str) -> Tuple[List[int] | torch.tensor | str]:
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        # logger.info(textlist)
        # logger.info(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones, bert.to(torch.float16 if is_half == True else torch.float32), norm_text


def get_spepc(hps: Dict, filename: str) -> torch.tensor:
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


def pack_audio(audio_bytes: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    # wav无法流式, 先暂存raw
    audio_bytes = pack_raw(audio_bytes, data, rate)

    return audio_bytes


def pack_raw(audio_bytes: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    audio_bytes.write(data.tobytes())

    return audio_bytes


def pack_wav(audio_bytes: BytesIO, rate: int) -> BytesIO:
    data = np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)
    wav_bytes = BytesIO()
    sf.write(wav_bytes, data, rate, format='wav')
    wav_bytes.seek(0)
    return wav_bytes


def cut_text(text: str, punc: str) -> str:
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text: str) -> bool:
    return not any(t.isalnum() or t.isalpha() for t in text)


def get_tts_wav(text: str,
                text_language: str,
                cut_punc: str = ",.;?!、，。？！；：…",
                ref_wav_path: str = "/root/code/ComfyChat/audio/wavs/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav",
                prompt_text: str = "疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？", 
                prompt_language: str = "zh",
                return_numpy: bool = False) -> Union[Tuple[np.ndarray, int], BytesIO]:
    t0 = ttime()
    if cut_punc == None:
        text = cut_text(text, default_cut_punc)
    else:
        text = cut_text(text, cut_punc)
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (is_half == True):
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])  # 将引导音频和初始的全零音频concat起来
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language.lower()]
    text_language = dict_language[text_language.lower()]
    phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)
    texts = text.split("\n")
    audio_bytes = BytesIO()  # 在内存中读写字节数据对象

    for text in texts:
        # 简单防止纯符号引发参考音频泄露
        if only_punc(text):
            continue

        audio_opt = []
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=gpt_config['inference']['top_k'],
                early_stop_num=hz * max_sec)
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if (is_half == True):
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
                                refer).detach().cpu().numpy()[0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        audio_bytes = pack_audio(audio_bytes, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16),
                                 hps.data.sampling_rate)
    # logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    print(hps.data.sampling_rate)
    return (hps.data.sampling_rate, np.frombuffer(audio_bytes.getvalue(), dtype=np.int16)) if return_numpy else pack_wav(audio_bytes, hps.data.sampling_rate)


if __name__ == '__main__':
    # wav_bytes = get_tts_wav('未开启TTS时，左键点击LLM生成的回复内容，返回本默认音频。', 'zh',)

    # # 将BytesIO对象保存为WAV文件
    # with open('/root/code/ComfyChat/audio/wavs/default.wav', 'wb') as f:
    #     f.write(wav_bytes.getbuffer())


    with open('/root/code/ComfyChat/server/config.ini', encoding='utf-8') as f:
        config = pytoml.load(f)


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


    def chatbot_selected2tts(text: str, text_language: str, cut_punc: str, gpt_path: str, sovits_path: str,
                             ref_wav_path: str, prompt_text: str, prompt_language: str) -> Tuple[int, np.ndarray]:
        global default_gpt_path, default_sovits_path
        if gpt_path != default_gpt_path and sovits_path != default_sovits_path:
            set_gpt_weights(gpt_path)
            set_sovits_weights(sovits_path)
            default_gpt_path = gpt_path
            default_sovits_path = sovits_path
        results = get_tts_wav(text, text_language, cut_punc, ref_wav_path, prompt_text, prompt_language, return_numpy=True)
        return results


    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gpt_sovits_voice = gr.Radio(["派蒙", "罗刹", "胡桃", "魈"], value="派蒙", label="Reference audio for GPT-SoVITS")
                cut_punc = gr.Textbox(value=",.;?!、，。？！；：…", label="Delimiters for GPT-SoVITS")
                lang = gr.Radio(["zh", "en"], value="zh", label="Language")
                gpt_path = gr.Textbox(value=default_gpt_path)
                sovits_path = gr.Textbox(value=default_sovits_path)
                ref_wav_path = gr.Textbox(value=os.path.join(parent_dir, config["gptsovits"]["wav"]["paimeng"]))
                prompt_text = gr.Textbox(value=config["gptsovits"]["prompt"]["paimeng"])
                prompt_language = gr.Textbox(value=config["gptsovits"]["language"]["paimeng"])
            with gr.Column(scale=11):
                text = gr.Textbox()
                audio = gr.Audio(type="numpy")
                button = gr.Button("Generate")

            gpt_sovits_voice.change(update_gpt_sovits, inputs=gpt_sovits_voice, outputs=[gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language])
            button.click(chatbot_selected2tts, inputs=[text, lang, cut_punc, gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language], outputs=audio)

    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True)