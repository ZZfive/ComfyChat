# comfyui-anime-seg
A Anime Character Segmentation node for comfyui, based on [this hf space](https://huggingface.co/spaces/skytnt/anime-remove-background) and forked from [abg-comfyui](https://github.com/kwaroran/abg-comfyui.git)
# Installation
1. git clone this repo to the custom_nodes directory
```
git clone https://github.com/LyazS/comfyui-anime-seg.git
```

2. Download dependencys on requirements.txt on comfyui
```
pip install -r requirements.txt
```

1. Download the `isnetis.onnx` model from [here](https://huggingface.co/skytnt/anime-seg/tree/main), and put it in the `custom_nodes/comfyui-anime-seg/models` directory.

# Usage
Create a "mask/Anime Character Seg" node, and connect the images to input, and it would segment the anime character and output the masks.
