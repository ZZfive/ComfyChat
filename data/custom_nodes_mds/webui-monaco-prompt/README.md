# WebUI Monaco Prompt

これは AUTOMATIC1111 氏の [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) と [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 用の拡張です。

プロンプトの編集をVSCodeでも使用されているエディタ実装 [Monaco Editor](https://microsoft.github.io/monaco-editor/) で行えるようにします。

## インストール

### AUTOMATIC1111 Stable Diffusion WebUI

`stable-diffusion-webui` の `Install from URL` からこのリポジトリのURL `https://github.com/Taremin/webui-monaco-prompt` を入力してインストールしてください。

### ComfyUI (Experimental)

下記の二通りからお好きな方法でインストールしてください。

1. `custom_nodes` にこのリポジトリを clone する
2. `ComfyUI Manager Menu` の `Install via Git URL` にこのリポジトリのURLを入力してインストールする

#### 以前のインストール方法

~~[Releases](https://github.com/Taremin/webui-monaco-prompt/releases) からzipファイルをダウンロードして `web/extensions` に展開してください。~~
v0.1.2からはこの方法ではインストールできなくなりました。

## 機能

- VIM キーバインディング対応 ([monaco-vim](https://github.com/brijeshb42/monaco-vim))
- 色付け機能
    - 標準表記
    - Dynamic Prompts拡張表記 
- オートコンプリート対応
    - デフォルトでは `danbooru.csv`, `extra-quality-tags.csv` を読み込んでいるので既に `a1111-sd-webui-tagcompete` を使用している方は違和感なく使えます
    - `<` を入力すると Extra Networks (HN/LoRA/LyCORIS) のみの候補を出せます
        - LyCORISは[a1111-sd-webui-lycoris](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris)拡張導入時に使用可能です

## 注意

### AUTOMATIC1111 Stable Diffusion WebUI

この拡張では標準のプロンプト編集で使用するtextareaを差し替えたり Extra Networks のリフレッシュへの対応などで、特定のHTML要素に依存したあまり汎用的でない手段を用いています。
そのため、HTML構造が変化したり既存機能の変更が行われた場合、利用できなくなることがあるかもしれません。
その場合は一時的に利用を中止することをおすすめします。

## その他

### 共通

ヘッダが邪魔な場合はエディタのコンテキストメニューから非表示にできます。（ヘッダで行える設定もコンテキストメニューから行えます。）

### AUTOMATIC1111 Stable Diffusion WebUI

設定はこの拡張のあるディレクトリの `settings` 内に保存されます。
認証未設定時は `global.json` 認証設定時は `user_[username].json` というファイル名です。

## クレジット

この拡張には [a1111-sd-webui-tagcomplete
](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete) のタグデータ(danbooru.csv, extra-quality-tags.csv)を同梱しています。

## ライセンス

[MIT](./LICENSE)
