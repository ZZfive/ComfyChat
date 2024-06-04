import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 连不上huggingface可以尝试打开此注释
os.environ["HF_HOME"] = r"D:\cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\cache"

import torch
import torchaudio
from IPython.display import Audio
import numpy as np

from ChatTTS import ChatTTS

chat = ChatTTS.Chat()
chat.load_models(compile=False) # 设置为True以获得更快速度


# 从ChatTTS.webui中获取
def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag):

    torch.manual_seed(audio_seed_input)  # 设置采样音色的随机种子
    rand_spk = chat.sample_random_speaker()  # 从高斯分布中随机采样出一个音色
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), text_data]


'''简单测试
torch要大于等于2.1.0，要安装pysoudfile，pip install pysoudfile，还需要conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing
'''
# texts = ["本文件中的信息仅供学术交流使用。其目的是用于教育和研究，不得用于任何商业或法律目的。作者不保证信息的准确性、完整性或可靠性。本文件中使用的信息和数据，仅用于学术研究目的。这些数据来自公开可用的来源，作者不对数据的所有权或版权提出任何主张。",]
# wavs = chat.infer(texts, use_decoder=True)
# torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)