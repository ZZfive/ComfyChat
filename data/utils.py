import os
import json


def list_subdirectories(path):
    subdirectories = []
    # 遍历指定路径下的所有文件和文件夹
    for entry in os.listdir(path):
        # 拼接子路径
        full_path = os.path.join(path, entry)
        # 如果是文件夹，则添加到列表中
        if os.path.isdir(full_path):
            subdirectories.append(entry)
    return subdirectories


# 用法示例
path = "/root/code/ComfyChat/data/custom_nodes_mds"
subdirectories_list = list_subdirectories(path)
print(len(subdirectories_list))
with open('/root/code/ComfyChat/data/geted_nodes.json', 'w') as f:
    json.dump(subdirectories_list, f, indent=4)

# with open('/root/code/ComfyChat/data/github_nodes.json', 'r') as f:
#     github_nodes = json.load(f)

# num = 0
# for node in subdirectories_list:
#     if node not in github_nodes:
#         print(node)
#         num += 1
# print(num)