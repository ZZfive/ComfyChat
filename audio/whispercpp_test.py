# from whispercpp import Whisper
# from whispercpp import api
# import numpy as np

# w = Whisper.from_pretrained("base")
# a = w.transcribe(np.ones((1, 16000))[0])
# print(a)

import whisperx

device = "cuda"
compute_type: str = 'float16'
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language='en',
                            download_root='/root/code/ComfyChat/weights')  # large-v2

# audio = whisperx.load_audio()