# ComfyUI-SizeFromPresets
![SizeFromPresets Preview](preview.png "SizeFromPresets Preview")
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)用のカスタムノードです。
- プリセットから選択したサイズの幅と高さを出力するノードを追加します。

## インストール手順
```
cd <ComfyUIがあるディレクトリ>/custom_nodes
git clone https://github.com/nkchocoai/ComfyUI-SizeFromPresets.git
```

## 追加されるノード
### Size From Presets (SD1.5)
- SD1.5用の画像サイズのプリセットを選択できます。
- 選択したサイズの幅と高さを出力します。
- プリセットは [presets/sd15.csv](presets/sd15.csv) から設定できます。

### Size From Presets (SDXL)
- SDXL用の画像サイズのプリセットを選択できます。
- 選択したサイズの幅と高さを出力します。
- プリセットは [presets/sdxl.csv](presets/sdxl.csv) から設定できます。

### Empty Latent Image From Presets (SD1.5)
- SD1.5用の画像サイズのプリセットを選択できます。
- 選択したサイズの空のLatent Imageと幅と高さを出力します。
- プリセットは [presets/sd15.csv](presets/sd15.csv) から設定できます。

### Empty Latent Image From Presets (SDXL)
- SDXL用の画像サイズのプリセットを選択できます。
- 選択したサイズの空のLatent Imageと幅と高さを出力します。
- プリセットは [presets/sdxl.csv](presets/sdxl.csv) から設定できます。

### Random ... From Presets (SD...)
- CSVファイルに記載したプリセットからランダムに選びます。
- Sizeの場合は、選ばれたサイズの幅と高さを出力します。
- Latent Imageの場合は、選ばれたサイズの空のLatent Imageと幅と高さを出力します。