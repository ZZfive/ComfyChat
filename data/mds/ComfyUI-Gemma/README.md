ComfyUI Gemma

## Install

https://www.kaggle.com/models/google/gemma

Export your Kaggle username and token to the environment:

```
export KAGGLE_USERNAME=fles123
export KAGGLE_KEY=cb8421571d5d0e29f0564b5a9b105405
```

Run python code to download models

```
import kagglehub
VARIANT='7b-it-quant' #@param ['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant']
weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')
print(f'{weights_dir}')
```

## Basic workflow

<img src="wf.png" raw=true>

https://github.com/chaojie/ComfyUI-Gemma/blob/main/workflow.json