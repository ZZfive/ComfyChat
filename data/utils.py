import os
import re
import json
import time
import logging

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
        json.dump(data, f, indent=4, ensure_ascii=False)


def load4json(path: str, default_value: Any = None) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.decoder.JSONDecodeError:
        print(f"Error: JSONDecodeError occurred while loading file: {path}")
        return default_value
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return default_value


def get_data_from_url(url: str) -> Any:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
       raise ValueError(f"Failed to get data from url {url}, err: {e}")
    

def parse_json(rsp):
    pattern = r'```json(.*)```'
    match = re.search(pattern, rsp, re.DOTALL)
    # print(match)
    # print(match.group(1))
    json_data = match.group(1) if match else ''
    return json.loads(json_data)


def create_logger(name: str, log_dir: str = '/root/code/ComfyChat/data/logs') -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = time.strftime("%Y%m%d:%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, f"{name}_{current_time}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__=='__main__':
    # path = "/root/code/ComfyChat/data/custom_nodes_mds"
    # subdirectories_list = list_subdirectories(path)
    # print(len(subdirectories_list))
    # save2json(subdirectories_list, "/root/code/ComfyChat/data/geted_nodes.json")

    rsp = '''
    ## Question and Answer Dataset for ComfyUI_Fictiverse Nodes in JSON Format:

    ```json
    [
        {
            "input": "What is the purpose of the ComfyUI_Fictiverse nodes?",
            "output": "The nodes in ComfyUI_Fictiverse extend the functionality of ComfyUI by providing custom image-processing algorithms."
        },
        {
            "input": "How do I install the ComfyUI_Fictiverse nodes?",
            "output": "Install ComfyUI, then clone or download the nodes to the `custom_nodes/ComfyUI_Fictiverse` folder."
        },
        {
            "input": "Which nodes are included in the ComfyUI_Fictiverse package?",
            "output": "The package includes Color Correction (obsolete), Displace Images with Mask, Add Noise to Image with Mask, and Displace Image with Depth nodes."
        },
        {
            "input": "What is the function of the Color Correction node?",
            "output": "The obsolete Color Correction node attempts to match the color of an image to a reference image."
        },
        {
            "input": "Explain the Displace Images with Mask node functionality.",
            "output": "It displaces images based on a directional mask, removing unwanted visual elements."
        },
        {
            "input": "Describe the purpose of the Add Noise to Image with Mask node.",
            "output": "It selectively adds noise to an image using a predefined mask."
        },
        {
            "input": "Can you briefly explain the functionality of the Displace Image with Depth node?",
            "output": "The Displace Image with Depth node utilizes image depth information to displace pixels of an image."
        }
    ]
    ```
    '''

    json_data = parse_json(rsp)
    print(type(json_data))
    print(json_data)