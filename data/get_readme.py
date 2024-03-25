import os
import json
from git import Repo

import requests


def get_repo_urls(json_url: str='https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json',
                  save_path: str='/root/code/ComfyChat/data/custom_node_list.json',
                  update: bool=False):
    try:
        if os.path.exists(save_path) and not update:
            with open(save_path, 'r', encoding='utf-8') as f:
                custom_node_list = json.load(f)
        else:
            response = requests.get(json_url)
            response.raise_for_status()
            custom_node_messages = response.json()['custom_nodes']

            custom_node_list = []
            for node in custom_node_messages:
                url = node['reference']
                if url.startswith("https://github.com") and url not in custom_node_list:
                    custom_node_list.append(url)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(custom_node_list, f, indent=4)

        return custom_node_list
    except Exception as e:
        print(e)

    # 出现异常时返回空列表
    return []


def clone_repos_and_extract_md_files(repo_urls, local_base_dir, save_base_dir):
    # 创建保存.md文件的目录
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)

    for i, repo_url in enumerate(repo_urls):
        save_dir = os.path.join(save_base_dir, repo_url.split('/')[-1])
        if os.path.exists(save_dir):
            continue

        # 克隆仓库到本地临时目录
        print(f"i: {i}, repo url: {repo_url}")
        local_repo_dir = os.path.join(local_base_dir, repo_url.split('/')[-1])
        Repo.clone_from(repo_url, local_repo_dir)

        # 提取.md文件并保存
        extract_md_files_from_local_repo(local_repo_dir, save_dir)


def extract_md_files_from_local_repo(repo_path, save_dir):
    # 创建保存.md文件的目录
    os.makedirs(save_dir, exist_ok=True)

    # 打开本地仓库
    repo = Repo(repo_path)

    # 获取仓库中的所有文件
    files = [item for item in repo.tree().traverse() if item.type == 'blob']

    for file in files:
        if file.path.endswith('.md'):
            # 保存.md文件
            file_content = file.data_stream.read().decode("utf-8")
            with open(os.path.join(save_dir, file.path.split('/')[-1]), 'w', encoding="utf-8") as f:
                f.write(file_content)


if __name__ == '__main__':
    # repo_urls = ['https://github.com/BlenderNeko/ComfyUI-docs']
    repo_urls = get_repo_urls()
    print(len(repo_urls))
    local_base_dir = '/root/code/ComfyChat/data/repos'
    save_base_dir = '/root/code/ComfyChat/data/mds'
    clone_repos_and_extract_md_files(repo_urls, local_base_dir, save_base_dir)