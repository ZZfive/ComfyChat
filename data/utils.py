import os
import json
from typing import List, Any

import requests


def list_subdirectories(path: str) -> List[str]:
    subdirectories = []
    # 遍历指定路径下的所有文件和文件夹
    for entry in os.listdir(path):
        # 拼接子路径
        full_path = os.path.join(path, entry)
        # 如果是文件夹，则添加到列表中
        if os.path.isdir(full_path):
            subdirectories.append(entry)
    return subdirectories


def save2json(data: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def load4json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_data_from_url(url: str) -> Any:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
       raise ValueError(f"Failed to get data from url {url}, err: {e}")


if __name__=='__main__':
    path = "/root/code/ComfyChat/data/custom_nodes_mds"
    subdirectories_list = list_subdirectories(path)
    print(len(subdirectories_list))
    save_json(subdirectories_list, "/root/code/ComfyChat/data/geted_nodes.json")