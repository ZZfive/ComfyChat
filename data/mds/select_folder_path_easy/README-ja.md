# select_folder_path_easy

## outputフォルダの指定を楽にする

![image01](/images/image01.png)

## このextensionは何ですか?
このextensionはノードを接続するだけで、生成した画像の出力先パスを管理しやすいパスに指定します。

## インストール方法
```
cd path/to/comfyui

cd custom_nodes

git clone https://github.com/Umikaze-job/select_folder_path_easy
```

# 使い方
1. \[Save Image\]ノードを右クリックして、\[Convert filename_prefix to input\]をクリックします。

![image02](/images/image02.png)

2. \[select folder path easy\]と\[Save Image\]を接続すればOKです。

## パラメータ説明
folder_name: フォルダ名

file_name: ファイル名を指定(\[file_name\]_\[画像ID\].pngというファイル名になります。)

time_format: 時間フォルダのフォーマットを指定します。