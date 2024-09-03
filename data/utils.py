import os
import re
import json
import logging
from datetime import datetime

from typing import List, Any, Tuple

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
    

def parse_json(rsp: str) -> Any:
    try:
        return json.loads(rsp)
    except Exception:
        pattern = r'```json(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        # print(match)
        # print(match.group(1))
        json_data = match.group(1) if match else ''
        return json.loads(json_data)


def create_logger(name: str, log_dir: str = '/root/code/ComfyChat/data/logs') -> logging.Logger:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # current_time = time.strftime("%Y%m%d:%H%M%S", time.localtime())
    # log_file = os.path.join(log_dir, f"{name}_{current_time}.log")
    log_file = os.path.join(log_dir, f"{name}.log")

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


def extract_name_extension(filepath: str) -> Tuple[str, str]:
    name_ext = os.path.basename(filepath)
    name, extension = os.path.splitext(name_ext)
    return name, extension


def get_latest_modification_time(directory: str) -> str:
    latest_time = None
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            # 获取文件的修改时间
            file_time = os.path.getmtime(filepath)
            # 更新最新的修改时间和文件名
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
    
    if latest_time:
        return datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return None


if __name__=='__main__':
    # path = "/root/code/ComfyChat/data/custom_nodes_mds"
    # subdirectories_list = list_subdirectories(path)
    # print(len(subdirectories_list))
    # save2json(subdirectories_list, "/root/code/ComfyChat/data/geted_nodes.json")

    rsp = '''
     Based on the provided information about ComfyUI_Fictiverse, here is a sample JSON format for the question and answer data pair:

    ```json
    [
        {
            "subject": "ComfyUI_Fictiverse",
            "question": "How do I install ComfyUI_Fictiverse?",
            "answer": "To install ComfyUI_Fictiverse, first install ComfyUI. Then, clone the repository into the 'custom_nodes' folder within your ComfyUI installation or download the ZIP and extract the contents to 'custom_nodes/ComfyUI_Fictiverse'."
        },
        {
            "subject": "Color Correction (obsolete)",
            "question": "What is the purpose of the Color Correction custom node in ComfyUI_Fictiverse?",
            "answer": "The Color Correction node in ComfyUI_Fictiverse attempts to match the color of an image to that of a reference image, although it is noted to be somewhat ineffective."
        },
        {
            "subject": "Displace Images with Mask",
            "question": "What does the Displace Images with Mask node do in ComfyUI_Fictiverse?",
            "answer": "The Displace Images with Mask node is a modified version of the WAS node in ComfyUI_Fictiverse. It displaces images based on a mask with directional amplitudes."
        },
        {
            "subject": "Add Noise to Image with Mask",
            "question": "How does the Add Noise to Image with Mask node work in ComfyUI_Fictiverse?",
            "answer": "The Add Noise to Image with Mask node in ComfyUI_Fictiverse applies noise to an image within a specified mask."
        },
        {
            "subject": "Displace Image with Depth",
            "question": "What is the function of the Displace Image with Depth node in ComfyUI_Fictiverse?",
            "answer": "The Displace Image with Depth node in ComfyUI_Fictiverse attempts to displace an image based on its depth."
        }
    ]
    ```

    This JSON format includes the subject, question, and answer for each of the custom nodes described in the documentation. Please note that the actual questions and answers should be tailored to cover all the relevant information in the document content.
    '''

    json_data = parse_json(rsp)
    print(type(json_data))
    print(json_data)