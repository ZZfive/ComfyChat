# select_folder_path_easy

## Easier specification of output folder

[日本語の説明はこちら](README-ja.md)

![image01](/images/image01.png)

## What is this extension?
This extension simply connects the nodes and specifies the output path of the generated images to a manageable path.

## How to install
```
cd path/to/comfyui

cd custom_nodes

git clone https://github.com/Umikaze-job/select_folder_path_easy
```

# Usage
1. right-click on the \[Save Image\] node and click on\[Convert filename_prefix to input\].

![image02](/images/image02.png)

2. connect \[select folder path easy\] and \[Save Image\] and you are good to go.

## Parameter Description
folder_name: Folder name

file_name: Specifies the file name (the file will be named "\[file_name\]_\[image_id\].png")

time_format: Specify the format of the time folder.
