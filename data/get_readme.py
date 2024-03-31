import os
import json
import shutil
from typing import List
from git import Repo

import requests


# 从设置默认的comfyui manager项目中提供的自定义节点信息json文件从抽取各个节点的repo url
def get_repo_urls(json_url: str='https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json',
                  save_path: str='/root/code/ComfyChat/data/custom_node_list.json',
                  update: bool=False) -> List:
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


# 将各个自定义节点repo git到本地后收集保存.md文件
def clone_repos_and_extract_md_files(repo_urls: List[str], local_base_dir: str, save_base_dir: str) -> None:
    # 创建保存.md文件的目录
    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)

    for i, repo_url in enumerate(repo_urls):
        save_dir = os.path.join(save_base_dir, repo_url.split('/')[-1])  # md保存路径
        local_repo_dir = os.path.join(local_base_dir, repo_url.split('/')[-1])  # 项目保存目录

        if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
            continue
        
        # 克隆仓库到本地临时目录
        print(f"i: {i}, repo url: {repo_url}")
        Repo.clone_from(repo_url, local_repo_dir)

        # 提取.md文件并保存
        extract_md_files_from_local_repo(local_repo_dir, save_dir)


# 搜集、保存.md文件的主要函数
def extract_md_files_from_local_repo(repo_path: str, save_dir: str) -> None:
    # 创建保存.md文件的目录
    os.makedirs(save_dir, exist_ok=True)

    # 打开本地仓库
    repo = Repo(repo_path)

    # 获取仓库中的所有文件
    files = [item for item in repo.tree().traverse() if item.type == 'blob']

    for file in files:
        if file.path.endswith('.md') or file.path.endswith('.mdx'):
            # 保存.md文件
            try:
                file_content = file.data_stream.read().decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_content = file.data_stream.read().decode("latin-1")
                except UnicodeDecodeError:
                    try:
                        file_content = file.data_stream.read().decode("ascii", errors="ignore")
                    except Exception as e:
                        print(f"Error decoding file '{file.path}': {e}")
                        continue
                    
            with open(os.path.join(save_dir, file.path.split('/')[-1]), 'w', encoding="utf-8") as f:
                f.write(file_content)


# 获取可能已收集过的的节点.md文件的保存路径名
def get_subdirs(directory: str) -> List[str]:
    subdirectories = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    return subdirectories


# 更新自定义节点repo中的.md文件，是针对新增的自定义节点repo，并不会对更新过的节点repo中的.md文件更新
def update_mds(save_base_dir: str, local_base_dir: str) -> None:
    custom_node_list = get_repo_urls(update=True)
    custom_node_name2urls = {repo_url.split('/')[-1]: repo_url for repo_url in custom_node_list}
    has_repos_names = get_subdirs(save_base_dir)
    not_has_repos = []
    
    for name, repo_url in custom_node_name2urls.items():
        if name not in has_repos_names:
            not_has_repos.append(repo_url)
            if os.path.exists(os.path.join(local_base_dir, name)):
                shutil.rmtree(os.path.join(local_base_dir, name))

    for name in has_repos_names:
        if len(os.listdir(os.path.join(save_base_dir, name))) == 0:
            not_has_repos.append(custom_node_name2urls[name])
            if os.path.exists(os.path.join(local_base_dir, name)):
                shutil.rmtree(os.path.join(local_base_dir, name))

    print(f"updated nodes {not_has_repos}, num of nodes {len(not_has_repos)}")
    clone_repos_and_extract_md_files(not_has_repos, local_base_dir, save_base_dir)


if __name__ == '__main__':
    # repo_urls = ['https://github.com/BlenderNeko/ComfyUI-docs']
    # repo_urls = get_repo_urls()
    # print(len(repo_urls))
    local_base_dir = '/root/code/ComfyChat/data/repos'
    save_base_dir = '/root/code/ComfyChat/data/mds'
    # clone_repos_and_extract_md_files(repo_urls, local_base_dir, save_base_dir)
    update_mds(save_base_dir, local_base_dir)