# ComfyUI-TextOnSegs
<img src='img/example_face.jpg' width='400'>
<img src='img/example_board.jpg' width='400'>  
  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)用のカスタムノードです。
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)の[Ultralytics Detector](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/detectors.md#ultralytics-detector)で検出したSEGSの範囲に、[ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)の[CR Draw Text](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-draw-text)でテキストを書き込むためのノードを追加します。

## インストール手順
```
cd <ComfyUIがあるディレクトリ>/custom_nodes
git clone https://github.com/nkchocoai/ComfyUI-TextOnSegs.git
```

### 前提となる拡張機能
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)

## 使い方
- (任意) 以下のフォルダにフォントファイル(*.ttf)を配置します。
  - `ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes/fonts`
  - 日本語などは文字化けするため、フォントファイルを追加する必要があります。

### 顔にテキストを書く
- [workflows/draw_text_on_face.json](workflows/draw_text_on_face.json) をD&Dで読み込みます。
- 「Draw Text」グループ内のTextノードやCalcMaxFontSizeノードなどの値を変更します。
- ワークフローを実行します。
  - 検出に失敗した場合、エラーが発生しますが仕様です。

### ボードにテキストを書く
- [Can't show this \(meme\) SDXL](https://civitai.com/models/293531) をダウンロードし、以下のフォルダに配置します。
  - `ComfyUI_windows_portable/ComfyUI/models/loras`
- [Board detector YOLO model \(For Can't show this \(meme\) SDXL\) \[Adetailer Model\] \- v1\.0](https://civitai.com/models/300228) をダウンロードし、以下のフォルダに配置します。
  - `ComfyUI_windows_portable/ComfyUI/models/ultralytics/bbox`
- [workflows/draw_text_on_board.json](workflows/draw_text_on_board.json) をD&Dで読み込みます。
- 「Draw Text」グループ内のTextノードやCalcMaxFontSizeノードなどの値を変更します。
- ワークフローを実行します。
  - 検出に失敗した場合、エラーが発生しますが仕様です。