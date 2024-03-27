# cd-tuner_negpip-ComfyUI
このカスタムノードは、[hako-mikan](https://github.com/hako-mikan)氏による以下の二つのwebUI拡張をComfyUI実装するものです。

+ https://github.com/hako-mikan/sd-webui-cd-tuner
：色調や書き込み量を調節する機能、とりあえず一部の機能のみ実装・・・
+ https://github.com/hako-mikan/sd-webui-negpip
：プロンプトにマイナスの重みを実装する機能

# 説明
loaderに二つのノードが追加されます。使い方の説明が必要なほど難しくないです。
cd-tunerのstart、endは0,1000の範囲で指定してください（step単位じゃないのは実装の都合です）。

# 謝辞
二つの実装の考案者である[hako-mikan](https://github.com/hako-mikan)氏に感謝いたします。
