# ComfyUI-TemporaryLoader
This is a custom node of ComfyUI that downloads and loads models from the input URL. The model is temporarily downloaded into memory and not saved to storage.

This could be useful when trying out models or when using various models on machines with limited storage.

## Installation
1. Use `git clone https://github.com/pkpkTech/ComfyUI-TemporaryLoader` in your ComfyUI custom_nodes directory
1. Use `pip install -r requirements.txt` in ComfyUI-TemporaryLoader directory

## Nodes
The "Load Checkpoint (Temporary)" and "Load LoRA (Temporary)" nodes will be added to the temporary_loader category.

In addition to the standard Load node, the following items are added
- ckpt_url: URL of the model
- ckpt_type: With `auto`, it looks at the file extension of the downloaded file.
- download_split: Specify the number of splits for parallel downloading.

In addition to these, a "Load Multi LoRA (Temporary)" node is also added.<br>
This is a node for loading multiple LoRAs.<br>
It can be loaded by URL or by specifying a filename in the same way as the standard LoRA Loader.<br>
Follow the format below

`{LoRA URL}` or `file:{LoRA file name}`<br>- e.g.) `https://example.com/anylora.safetensors` or `file:anylora.safetensors`

`{strength_model}:{strength_clip}:{LoRA URL} or file:{LoRA file name}`<br>- e.g.) `0.4:1.0:https://example.com/anylora.safetensors`

`{strength_model}:{strength_clip}:{ckpt_type}:{LoRA URL} or file:{LoRA file name}`<br>- e.g.) `0.4:1.0:other:https://example.com/anylora.pt`

LoRAs for which strength_model, strength_clip and ckpt_type are not specified in text will reflect the node's set values.<br>
You can also specify only a part.<br>- e.g.) `0.1::https://example.com/anylora.safetensors` (strength_model is 0.1, strength_clip and ckpt_type follow node settings).

To avoid confusion as to what LoRA it is, you can also write a comment.
Comments should start the line with `#`.

e.g.)
```
#thatLoRA
https://example.com/anylora.safetensors

#The LoRA of the super landscape
0.3::https://example.com/superlora.safetensors

#The usual LoRA
0.5:0.8:file:favorite_lora_v6.safetensors
```
