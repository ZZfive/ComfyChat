# ComfyUI Slothful Attention & Near-sighted Attention

This custom node allow controlling output without training.


# Update history

 - 2023-12-16: Add `Near-sighted Tile` and `Near-sighted Attention`
 - 2023-12-11: Release

## What's this?

### Slothful Attention

This nodes allow controlling output images by reducing K and V samples on self-attentions.

The reducing method is similar to [Spatial-Reduction Attention](https://paperswithcode.com/method/spatial-reduction-attention),
but generating speed may not be increased on typical image sizes due to overheads. (In some cases, slightly slower) 

### Near-sighted Tile

Like HyperTile, this nodes split samples as Q for self-attention. And serves K, V that concatinated local and global samples by given ratio.

This may improve details of images.

### Near-sighted Attention

Near-sighted Tile + Slothful Attention.


# 概要

### Slothful Attention

セルフアテンションの K,V を削減することで、画像のコントロールを行います。

[Spatial-Reduction Attention](https://paperswithcode.com/method/spatial-reduction-attention) に近いですが、
これ単体では速度向上やメモリ削減などの効果は薄いです。

基本的には画像コントロール用の機能と考えてください。

### Near-sighted Tile

HyperTileのように、サンプルを分割して self-attentionの Q として利用します。
また、このときローカルとグローバルを指定された比率で結合して K, V として利用します。

これにより、画像のディテールが改善するかもしれません。

### Near-sighted Attention

Near-sighted Tile と Slothful Attention の両方の機能を持ったノードです。

設定項目は多いですが、速度向上とある程度の画質コントロールが可能となります。


## Performance

UNet speed only. (not includes model loading, vae decoding etc.)

| node | 512x512 | 1024x1024 |
|:----:|----:|----:|
| Slothful Attention | -8.5% | +28.7% |
| Near-sighted Tile | -11.3% | +49.2% |
| Near-sighted Attention | -9.4% | +57.4% |

UNet速度のみの比較です。
大きめ画像サイズならば速度が上がります。

## Usage

Insert this nodes into MODEL connection before sampler. 

[sample workflow](workflow_sa.json)

![](images/workflow_sa.png)


Conjunction with hypernetworking and token merging May cause some problems. (These will patch attentions too) 

Tested with SD1.5, SDXL and SSD-1B based models. With LCM-lora, Lora, Controlnet or IPAdapter seems okey. 



## 使い方


サンプラーに入力するMODELの前にこのノードを差し込んでください

アテンションの Q,K,V にパッチを入れている関係で、Hypernetwork, TOME などの併用は問題出るかもしれません。

SD1.5, SDXL, SSD-1B ベースモデルで確認しています。LCM-lora, lora, Controlnet, IPAdapter などは使えてそうです。


## Tips

### Slothful Attention

`in_...` and `out_..` parameters: individual parameters for `in` and `out` blocks

Slothful is reduction rate. (this will be decreased by depth_decay and time_decay)

You can set another blend ratio for K, V

Bigger in_k_blend may reduce noises.


### Near-sighted Tile

tile_size is in latent space, so tile_size: 64 is 512x512px.

Smaller tile_size may improve image details, but may break consistency of image.

Larger global_ratio may prevent breaking consistency, but decrease details.

### Near-sighted Attention

Tiling is same to Near-sighted tile.

Low tile_size and high slothful may cause lack K and V. This will decrease quarity.

## パラメータ

### Slothful Attention

 - time_decay: ステップが進むごとに効果を弱める係数です
 - keep_middle: Trueのとき、middleブロックには適用しません
 
`in_...` `out_..` パラメータ: inブロック, outブロックに別のパラメータを適用できます

Slothful（を depth_decay, time_decayで減らした値）が削減比率になります。

time_decayについては、peak_time (開始が0、終了が1)のステップでは軽減無しで、そこから離れると time_decayに従って効果が軽減されます。
構図への影響を弱めたいときは、peak_time:0.5 time_decay: 2.5 などの設定が良いかもしれません
出力の品質が悪いときは、time_decayを上げるか、peak_timeを下げてみてください。
（最後の方のstepで影響を減らす目的です）

削減時は n サンプルごとに 1 サンプル取り出す one と、n サンプルを mode によってプーリングする pool を
ブレンドします。ブレンド率は K, V で別の値を指定出来ます。

avr（nサンプルの平均。ぼかしたような感じです） max（max_pooling。シャープネスに近いかもしれません）をブレンドすることで
アテンションの結果を連続的に変化させられます。

基本的には、ブレンド率は0.5以下が良いかと思います。

ブレンド率上げたときの画像変化は状況によって違うのですが、影響が大きいのは in_k_blend と out_v_blend です

in_k_blend を上げると、髪の毛の描画などが細かくなることもあります。
i2iとして使うときは、入力の細部への注目が弱くなるのかノイズや細かいパーツの影響が弱くなるようでした。

in_mode による変化はそこまで大きくないですが、1D系だと陰影や光沢などが軽視される傾向があるようです。

in_v_blendは モードによる違いが出やすいみたいです。
AVGの場合は輪郭が不明瞭になったりします。服の模様とかは結構影響受けやすいみたいです。
MAXの場合はAVGよりドラスティックな変化になります。モデルによりますが絵画的な描画になることもあります。

in_k_blendを上げると、服の模様などが不鮮明になったり、被写界深度（ボケ）っぽい効果になったりするようです。
絵がくっきりしすぎている場合はここを調整すると良い具合になってくれることもあります。

out_v_blend はコントラストやシャープネスに関係するようです。
AVG系モードでは柔らかめ、コントラスト低めの出力、MAX系モードでは固め、コントラスト高めの出力の傾向が出ます。

1Dの方がより強く効果が出ますが、2Dに比べて描画が崩れやすい傾向にあるようです。


### Near-sighted Tile

tile_sizeは潜在空間での寸法です。なので、tile_size: 64 はピクセルでは 512x512px になります。
（SD1.5の最も浅い層では）

tile_sizeを下げるとディテールが改善するかもしれませんが、画像の一貫性は損なわれやすいです。

global_ratioを上げると一貫性は保たれやすいですが、ディテールは低下するかも知れません。


### Near-dighted Attention

タイル分割については、Near-sighted Tile と同等ですが、base・peakで別の値が指定出来るようになっています。
peak_global_ratioを高めに設定しておくと、一貫性が確保しやすいみたいです。


注意として、低いtile_sizeと 高いslothfulの組み合わせでは K, V がかなり少なくなるため、
品質が目に見えて悪くなります。


## Output sample

Belows are generated on same parameter, except sloth attention.
ｓ
### SSD-1b based model with lcm-lora

 - checkoint: [ssd-1b-animagine](https://huggingface.co/furusu/SSD-1B-anime)
 - lcm-lora: [lcm-animagine](https://huggingface.co/furusu/SD-LoRA/blob/main/lcm-animagine.safetensors)

| Without | With Slothful Attention |
|----|----|
| ![](images/ssd1b_lcm.webp) | ![](images/ssd1b_lcm_mix.webp) |

more sample images: [pooling_modes.md](pooling_modes.md), [tiling_sizes.md](tiling_sizes.md)

