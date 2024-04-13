# ComfyUI_aspect_ratios

[sd-webui-ar](https://github.com/alemelis/sd-webui-ar?tab=readme-ov-file)をもとにComfyUI用のアスペクト比セレクターを作りました

![image](img1.png)

# 使用法

Empty Latent ImageをAspect Ratios Nodeに置き換える

- `size`：基準となるサイズ
- `aspect_ratios`：アスペクト比を設定
- `standard`：サイズの基準を幅にするか高さにするか
- `swap_aspect_ratio`：アスペクト比を入れ替える（1:2 -> 2:1に変える）
- `batch_size`：画像の作成枚数

## 構成

設定ファイル`aspect_ratios.txt`が`ComfyUI\custom_nodes\ComfyUI_aspect_ratios`に作成されます。

アスペクト比はファイルで設定できます。<br>
例
```aspect_ratios.txt
1:1, 1/1 # 1:1 ratio based on minimum dimension
3:2, 3/2 # Set width based on 3:2 ratio to height
4:3, 4/3 # Set width based on 4:3 ratio to height
16:9, 16/9 # Set width based on 16:9 ratio to height
1.618:1, 1.618/1
# 1.414:1, 1.414/1
```
- 先頭の`1:1,`はUIに表示されるもの
- 次の`1/1`は内部で処理される値
    - ※`数字`と`/`以外は含めない
- 最後に`# 1:1 ratio based on minimum dimension`はコメント

※最初の行に`#`がある場合は読み取られません

## 使用例

![image](img2.png)

`standard`をwidthに設定すると、幅が基準となり`幅が1024`、高さが512になります。

---

![image](img3.png)

`standard`をheightに設定すると、高さが基準となり`高さが1024`、幅が2048になります。


# アスペクト比の見方

アスペクト比は、幅と高さの比率を表す数字の組み合わせです。<br>
例えば「16:9」のアスペクト比は幅が16、高さが9の比率を指しています。

シンプルな見方「16:9 = 幅:高さ」