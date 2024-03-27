# ComfyUI Prompt ExtraNetworks

これは [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 用のカスタムノードです。
`LoraLoader` や `HypernetworkLoader` の代わりにプロンプトを受け取って、プロンプト内の指定によって LoRA や HN を読み込み適用します。
このカスタムノードの主目的は、プロンプトがランダムに変更されるときなどに `LoraLoader` ノードを繋ぎ直さずに変更できるようにすることです。

## 機能

### LoRA

以下の文法をプロンプトに含めることでLoRAを読み込みます。

```
<lora:lora-filename[:model_strength[:clip_strength]]>
```

- lora-filename: LoRAのファイル名。拡張子も含めて指定してください。(LoraLoaderで指定するファイル名と同じ)
- model_strength: モデルへの適用率。省略した場合は `1.0` になります。
- clip_strength: CLIPへの適用率。省略した場合は `model_strength` と同じになります。

#### LoRA Block Weight

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) をインストールしている場合、以下のように書くことで LoRA Block Weight を使用することが出来ます。

```
<lora:lora-filename[:model_strength[:clip_strength]][:lbw=lbw_preset]>
```

- lbw_preset: `SDXL-ALL` などのプリセット名か `1,0,0,0,0,0,1,1,1,1,1,1` など直接ウェイト表記
  - ComfyUI-Inspire-Pack で使用できるプリセット名が使用できます

### HyperNetwork

以下の文法をプロンプトに含めることでHyperNetworkを読み込みます。

```
<hypernet:hn-filename[:strength]>
```

- hn-filename: HyperNetworkのファイル名。拡張子も含めて指定してください。（HypernetworkLoaderで指定するファイル名と同じ）
- strength: モデルへの適用率。

**この機能はテストされていません。**

## ライセンス

[MIT](./LICENSE)
