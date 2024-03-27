# Introduction
Some nodes for stable diffusion comfyui.Sometimes it helps conveniently to use less nodes for doing the same things.

If you use workflow in my "blogs" repo, you need to dowmload these nodes.I don't guarantee that the nodes will stay the same always. Some nodes maybe have been changed if you update the new version.
# How to install
## The repo
The same with others custom nodes. Just cd custom_nodes and then git clone.
## Translator model
If you use prompt translator to translate Chinese to English offline, you need download some models.
Download the translator models from https://huggingface.co/facebook/mbart-large-50-many-to-one-mmt/tree/main into folder named "model" of this repo.
The model folder tree of this repo:
model/
└── mbart-large-50-many-to-many-mmt__only_to_English/
    ├── pytorch_model.bin
    ├── config.json
    ├── sentencepiece.bpe.model
    ├── special_tokens_map.json
    ├── tmp2l0rt359
    └── tokenizer_config.json
## Environments
cd (this repo)
pip install -r requirements.txt
# Thanks
Some codes are from The official [ComfyUI](https://github.com/comfyanonymous/ComfyUI.git) and other custom nodes like The [was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui.git).
The translator's main code is from [prompt_translator](https://github.com/ParisNeo/prompt_translator.git).