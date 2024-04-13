
# ComfyUI-TemporaryLoader
入力されたURLからモデルをダウンロードして読み込むComfyUIのカスタムノードです。

モデルは一時的にメモリにダウンロードされ、ストレージには保存されません。

## Installation
1. ComfyUIのcustom_nodesディレクトリで `git clone https://github.com/pkpkTech/ComfyUI-TemporaryLoader` を実行
1. ComfyUI-TemporaryLoaderディレクトリに移動して `pip install -r requirements.txt` を実行

## Nodes
"temporary_loader" カテゴリに、"Load Checkpoint (Temporary)" と "Load LoRA (Temporary)" ノードが追加されます。

標準のLoadノードに以下の項目が追加されています。
- ckpt_url: 使用するモデルのURL。
- ckpt_type: モデルの種類。`auto`でダウンロードしたファイルの拡張子で自動的に判断されますが、ファイル名が取得できない場合もあるので、その時に`safetensors`か`ohter`かを手動で選択してください。
- download_split: 並列ダウンロードの分割数。

これらに加えて、"Load Multi LoRA (Temporary)"ノードも追加されます。<br>
これは、複数のLoRAを読み込むためのノードです。<br>
URL、または標準のLoRA Loaderと同じようにファイル名を指定して読み込みます。<br>
下記のフォーマットに従って記述してください

`{LoRA URL}` or `file:{LoRA file name}`<br>- 例) `https://example.com/anylora.safetensors` or `file:anylora.safetensors`

`{strength_model}:{strength_clip}:{LoRA URL} or file:{LoRA file name}`<br>- 例) `0.4:1.0:https://example.com/anylora.safetensors`

`{strength_model}:{strength_clip}:{ckpt_type}:{LoRA URL} or file:{LoRA file name}`<br>- 例) `0.4:1.0:other:https://example.com/anylora.unknownext`

strength_model、strength_clip、ckpt_typeをテキストで指定していないLoRAには、ノードの設定値が反映されます。<br>
一部のみ指定することもできます。<br>- 例) `0.1::https://example.com/anylora.safetensors` (strength_modelが0.1になり、strength_clipとckpt_typeはノードの設定に従う)

何のLoRAか忘れないようにコメントを書くこともできます。<br>
コメントは行頭を`#`にしてください。

例)
```
#あのLoRA
https://example.com/anylora.safetensors

#すごそうなLoRA
0.3::https://example.com/superlora.safetensors

#いつものLoRA
0.5:0.8:file:favorite_lora_v6.safetensors
```
