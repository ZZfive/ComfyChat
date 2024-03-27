# LCMSampler-ComfyUI

この拡張ノードは[SSD-1B-anime](https://huggingface.co/furusu/SSD-1B-anime)のLCM化LoRAを使うためのものです。本家LCMでの動作は保証しません（そもそも現バージョンでは重みをロードできない）。

LCMによる高速生成を生かすため、デコーダとしてTAESDを利用するためのノードも用意しています。こちらは[ComfyUI-OtherVAEs](https://github.com/M1kep/ComfyUI-OtherVAEs)を参考にしています。

# 使い方
通常のKSamplerではなく、カスタムサンプラー対応のやつを使います。`sampling/custom_sampling/Load SamplerLCM`からサンプラーを引っ張って繋げてください。

TAESD(`loader/Load TAESD`)はロードするとVAEと同じように使えるようになるはず。VRAMが足りなかったらmax_batch_sizeを小さくしてください。

詳しくはワークフローで、以下からチェックポイントと二つのLoRAをダウンロードする必要があります。

モデル：https://huggingface.co/furusu/SSD-1B-anime

# credit
https://github.com/luosiallen/latent-consistency-model

https://github.com/M1kep/ComfyUI-OtherVAEs
