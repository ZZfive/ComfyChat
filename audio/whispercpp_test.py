from whispercpp import Whisper
from whispercpp import api
import numpy as np

w = Whisper.from_pretrained("base")
a = w.transcribe(np.ones((1, 16000))[0])
print(a)