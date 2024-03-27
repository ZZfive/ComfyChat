# ComfyUI-PromptUtilities
![PromptUtilities Preview](preview.png "PromptUtilities Preview")  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)用のカスタムノードです。
- プロンプト周りの便利なノードを追加します。

## インストール手順
```
cd <ComfyUIがあるディレクトリ>/custom_nodes
git clone https://github.com/nkchocoai/ComfyUI-PromptUtilities.git
```

## 追加されるノード
### Join String List (実験中)
- 入力として受け取った `argN` を `separator` で結合した文字列を出力します。
- 動作確認が十分にできていないので、バグがあるかもしれません。

![Example Join String List](img/ex_join.png "Example Join String List")  

### Format String (実験中)
- 入力として受け取った `argN` を `prompt` に埋め込んだ文字列を出力します。
- `prompt` において、 `[N]` は `argN` の値に置き換わります。
- 動作確認が十分にできていないので、バグがあるかもしれません。

![Example Format String](img/ex_format.png "Example Format String")  

### Load Preset
- 選択したプリセットのプロンプト（文字列）を出力します。
- プリセットは [presets](presets) ディレクトリ内に配置されたCSVファイルに記載します。
- [Easy Prompt Selector](https://github.com/blue-pen5805/sdweb-easy-prompt-selector?tab=readme-ov-file#customization) のymlファイルにも一部対応しています。

![Example Load Preset](img/ex_preset.png "Example Load Preset")

### Load Preset (Advanced)
- 選択したプリセットの以下の値を出力します。
  - ポジティブプロンプト
  - ネガティブプロンプト
  - LoRAとその強度
  - LoRA Stack　([Efficiency Nodes](https://github.com/jags111/efficiency-nodes-comfyui)用)
- プリセットは [advanced_presets](advanced_presets) ディレクトリ内に配置されたJSONファイルに記載します。

![Example Load Preset Advanced 01](img/ex_preset_adv_01.png "Example Load Preset Advanced 01")
![Example Load Preset Advanced 02](img/ex_preset_adv_02.png "Example Load Preset Advanced 02")

### Random Preset / Random Preset (Advanced) (実験中)
- 選択したファイル内からランダムに選ばれたプリセットの値を出力します。
- 動作確認が十分にできていないので、バグがあるかもしれません。

### Const String
- 入力した文字列を出力します。

### Const String(multi line)
- 入力した文字列を出力します。
- 複数行で入力できます。

## その他
- [config.ini.example](config.ini.example) を `config.ini` に名前を変更することで、 [presets](presets) ディレクトリ内に配置されたプリセットから Wildcard 形式のテキストファイルを `output_csv_presets_as_wildcards` で指定したディレクトリに出力します。

## おすすめの拡張機能
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
  - [Preset Text](https://github.com/pythongosssss/ComfyUI-Custom-Scripts?tab=readme-ov-file#preset-text) : 「Load Preset」ノードのようにテキストのプリセット読み込みができる。ComfyUI上でプリセットを保存できる。
  - [Show Text](https://github.com/pythongosssss/ComfyUI-Custom-Scripts?tab=readme-ov-file#show-text) : 入力として受け取った文字列を表示する。
  - [String Function](https://github.com/pythongosssss/ComfyUI-Custom-Scripts?tab=readme-ov-file#string-function) : 文字列の追加や置換などを行う。
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
  - [Wildcard](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/ImpactWildcard.md) : テキストファイルからランダムなプロンプトを選択する。
- [UE Nodes](https://github.com/chrisgoringe/cg-use-everywhere)
  - Anything Everywhere : 入力した値を他のノードの未接続の入力に出力する。[img/ex_preset_adv_01.png](img/ex_preset_adv_01.png) で使用。
