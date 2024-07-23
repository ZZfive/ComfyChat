#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/06/21 21:48:36
@Author  :   zzfive 
@Desc    :   None
'''

import os
import sys
import json
import hashlib
import datetime
import random
from typing import List, Any, Tuple
import logging
from logging import StreamHandler, Formatter
from logging.handlers import RotatingFileHandler

import fitz
import textract
import pandas as pd
from bs4 import BeautifulSoup

# 初始共用的filehander
if not os.path.exists('./logs'):
    os.mkdir('./logs')
    # 日志保存在app.log中，设置日志等级和格式
# time_flag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
f_handler = RotatingFileHandler(f'./logs/ComfyChat.log', maxBytes=100*1024*1024, backupCount=7, encoding='utf-8')
f_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
f_handler.setLevel(logging.INFO)


def create_logger(name: str) -> logging.Logger:
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 输出到控制台
    c_handler = StreamHandler(sys.stdout)
    c_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    
    logger.addHandler(f_handler)
    
    return logger


logger = create_logger("embedding_extract")


class FileName:
    """Record file original name, state and copied filepath with text format."""

    def __init__(self, root: str, filename: str, _type: str) -> None:
        self.root = root
        self.prefix = filename.replace('/', '_')
        self.basename = os.path.basename(filename)
        self.origin = os.path.join(root, filename)
        self.copypath = ''
        self._type = _type  # 文件类型
        self.state = True  # 是否抽取成功
        self.reason = ''

    def __str__(self) -> str:
        return '{},{},{},{}\n'.format(self.basename, self.copypath, self.state,
                                      self.reason)


class FileOperation:
    """Encapsulate all file reading operations."""

    def __init__(self) -> None:
        self.image_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
        self.md_suffix = ['.md', '.mdx']
        self.text_suffix = ['.txt', '.text']
        self.excel_suffix = ['.xlsx', '.xls', '.csv']
        self.pdf_suffix = '.pdf'
        self.ppt_suffix = '.pptx'
        self.html_suffix = ['.html', '.htm', '.shtml', '.xhtml']
        self.word_suffix = ['.docx', '.doc']
        # 除图片后缀的所有文本文件后缀
        self.normal_suffix = self.md_suffix + self.text_suffix + self.excel_suffix + [
                                  self.pdf_suffix
                              ] + self.word_suffix + [self.ppt_suffix
                                                      ] + self.html_suffix

    # 返回文件所属类型
    def get_type(self, filepath: str) -> str:
        filepath = filepath.lower()
        if filepath.endswith(self.pdf_suffix):
            return 'pdf'

        if filepath.endswith(self.ppt_suffix):
            return 'ppt'
        
        for suffix in self.md_suffix:
            if filepath.endswith(suffix):
                return 'md'

        for suffix in self.image_suffix:
            if filepath.endswith(suffix):
                return 'image'

        for suffix in self.text_suffix:
            if filepath.endswith(suffix):
                return 'text'

        for suffix in self.word_suffix:
            if filepath.endswith(suffix):
                return 'word'

        for suffix in self.excel_suffix:
            if filepath.endswith(suffix):
                return 'excel'

        for suffix in self.html_suffix:
            if filepath.endswith(suffix):
                return 'html'
        return None

    # 计算文件
    def md5(self, filepath: str) -> str:
        hash_object = hashlib.sha256()
        with open(filepath, 'rb') as file:
            chunk_size = 8192  # 8 KB
            while chunk := file.read(chunk_size):
                hash_object.update(chunk)  # 将每个chunk更新到hash_object中，防止文件过大导致内存不足

        return hash_object.hexdigest()[0:8]  # 最终返回整个文件sha256值的前八位

    # 对所有文件的处理情况进行记录
    def summarize(self, files: list) -> None:
        success = 0
        skip = 0
        failed = 0

        for file in files:
            if file.state:
                success += 1
            elif file.reason == 'skip':
                skip += 1
            else:
                logger.info('{} {}'.format(file.origin, file.reason))
                failed += 1

            logger.info('{} {}'.format(file.reason, file.copypath))
        logger.info('累计{}文件，成功{}个，跳过{}个，异常{}个'.format(len(files), success,
                                                      skip, failed))

    # 扫描指定路径下的所有文件
    def scan_dir(self, repo_dir: str) -> List[FileName]:
        files = []
        for root, _, filenames in os.walk(repo_dir):
            for filename in filenames:
                _type = self.get_type(filename)
                if _type is not None:
                    files.append(
                        FileName(root=root, filename=filename, _type=_type))
        return files

    def read_pdf(self, filepath: str) -> str:
        # load pdf and serialize table

        text = ''
        with fitz.open(filepath) as pages:
            for page in pages:
                text += page.get_text()
                tables = page.find_tables()
                for table in tables:
                    tablename = '_'.join(
                        filter(lambda x: x is not None and 'Col' not in x,
                               table.header.names))
                    pan = table.to_pandas()
                    json_text = pan.dropna(axis=1).to_json(force_ascii=False)
                    text += tablename
                    text += '\n'
                    text += json_text
                    text += '\n'
        return text

    def read_excel(self, filepath: str) -> Any:
        table = None
        if filepath.endswith('.csv'):
            table = pd.read_csv(filepath)
        else:
            table = pd.read_excel(filepath)
        if table is None:
            return ''
        json_text = table.dropna(axis=1).to_json(force_ascii=False)
        return json_text

    def read(self, filepath: str) -> Tuple[str, Any]:
        file_type = self.get_type(filepath)

        text = ''

        if not os.path.exists(filepath):
            return text, None

        try:

            if file_type == 'md' or file_type == 'text':
                with open(filepath) as f:
                    text = f.read()

            elif file_type == 'pdf':
                text += self.read_pdf(filepath)

            elif file_type == 'excel':
                text += self.read_excel(filepath)

            elif file_type == 'word' or file_type == 'ppt':
                # https://stackoverflow.com/questions/36001482/read-doc-file-with-python
                # https://textract.readthedocs.io/en/latest/installation.html
                text = textract.process(filepath).decode('utf8')
                if file_type == 'ppt':
                    text = text.replace('\n', ' ')

            elif file_type == 'html':
                with open(filepath) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text += soup.text

        except Exception as e:
            logger.error((filepath, str(e)))
            return '', e
        text = text.replace('\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('  ', ' ')
        text = text.replace('  ', ' ')
        text = text.replace('  ', ' ')
        return text, None
    

# 追踪记录query及生成对应答案，当此类的示例被销毁时，会将记录的内存写入示例初始化时设置的文件路径中
class QueryTracker:
    """A class to track queries and log them into a file.

    This class provides functionality to keep track of queries and write them
    into a log file. Whenever a query is made, it can be logged using this
    class, and when the instance of this class is destroyed, all logged queries
    are written to the file.
    """

    def __init__(self, log_file_path) -> None:
        """Initialize the QueryTracker with the path of the log file."""
        self.log_file_path = log_file_path
        self.log_list = []

    def log(self, key: str, value: str = '') -> None:
        """Log a query.

        Args:
            key (str): The key associated with the query.
            value (str): The value or result associated with the query.
        """
        self.log_list.append((key, value))

    def __del__(self) -> None:
        """Write all logged queries into the file when the QueryTracker
        instance is destroyed.

        It opens the log file in append mode, writes all logged queries into
        the file, and then closes the file. If any exception occurs during this
        process, it will be caught and printed to standard output.
        """
        try:
            with open(self.log_file_path, 'a', encoding='utf8') as log_file:
                for key, value in self.log_list:
                    log_file.write(f'{key}: {value}\n')
                log_file.write('\n')
        except Exception as e:
            print(e)


def histogram(values: list):
    """Print histogram log string for values."""
    values.sort()
    _len = len(values)
    if _len <= 1:
        return ''

    median = values[round((_len - 1) / 2)]
    _sum = 0
    min_val = min(values)
    max_val = max(values)
    range_width = max(1, round(0.1 * (max_val - min_val)))
    ranges = [(i * range_width, (i + 1) * range_width)
              for i in range((max_val // range_width) + 1)]

    # 计算每个范围的数值总数
    total_count = len(values)
    range_counts = [0] * len(ranges)
    for value in values:
        _sum += value
        for i, (start, end) in enumerate(ranges):
            if start <= value < end:
                range_counts[i] += 1
                break

    range_percentages = [(count / total_count) * 100 for count in range_counts]

    log_str = 'length count {}, avg {}, median {}\n'.format(
        len(values), round(_sum / len(values), 2), median)
    for i, (start, end) in enumerate(ranges):
        log_str += f'{start}-{end}  {range_percentages[i]:.2f}%'
        log_str += '\n'
    return log_str


def add_first_title(md_path: str, repo_name: str) -> None:
    heading = f'# {repo_name}\n\n'

    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            content = file.read()

         # 在最前面添加新的一级标题
        new_content = heading + content

        # 将新内容写回文件
        with open(md_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
    except FileNotFoundError:
        print(f"文件 '{md_path}' 未找到")
    except PermissionError:
        print(f"没有权限修改文件 '{md_path}'")
    except Exception as e:
        print(f"修改文件时出错: {e}")


def clean_community_docs(source: str, base_dir: str) -> None:
    if source == "ComfyUI-docs":  # 删除每个子路径中的index.md文档
        for item in os.listdir(base_dir):
            temp_path = os.path.join(base_dir, item)
            if os.path.isdir(temp_path):
                for iitem in os.listdir(temp_path):
                    if iitem == "index.md":
                        ttemp_path = os.path.join(temp_path, iitem)
                        os.remove(ttemp_path)
            else:
                if item == "index.md":
                    os.remove(temp_path)
    elif source == "SaltAI-Web-Docs":
        for item in os.listdir(base_dir):
            temp_path = os.path.join(base_dir, item)
            if os.path.isdir(temp_path):
                for iitem in os.listdir(temp_path):
                    if iitem == "licenses.md":
                        md_path = os.path.join(temp_path, iitem)
                        add_first_title(md_path, item)
    elif source == "comfyui-nodes-docs":
        pass
    elif source == "comflowy":
        pass
    else:
        raise ValueError("Wrong Community docs")
    

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
    

def save2json(data: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def random_sample_question(alpaca_path: str = "/root/code/ComfyChat/data/message_jsons/v1/alpaca_gpt4_data_modification.json",
                           data_path: str = "/root/code/ComfyChat/data/message_jsons/v2/community_en.json",
                           can_questions_path: str = "/root/code/ComfyChat/server/source/questions/en/can_questions.json",
                           cannot_questions_path: str = "/root/code/ComfyChat/server/source/questions/en/cannot_questions.json"
                           ) -> None:
    random.seed(42)
    can_questions = load4json(can_questions_path, [])
    cannot_questions = load4json(cannot_questions_path, [])

    can_all = load4json(data_path)
    cannot_all = load4json(alpaca_path)

    can_samples = random.sample(can_all, 100)
    cannot_samples = random.sample(cannot_all, 65)

    for sample in can_samples:
        can_questions.append(sample["messages"][0]["content"])

    for sample in cannot_samples:
        cannot_questions.append(sample["messages"][0]["content"])

    save2json(can_questions, can_questions_path)
    save2json(cannot_questions, cannot_questions_path)


if __name__ == "__main__":
    # repo_dir = "/root/code/ComfyChat/server/source"
    # fo = FileOperation()
    # files = fo.scan_dir(repo_dir)
    # for file in files:
    #     print(file.origin)

    # source = "SaltAI-Web-Docs"
    # base_dir = "/root/code/ComfyChat/server/source/knowledges/SaltAI-Web-Docs/md"
    # clean_community_docs(source, base_dir)

    # md_path = "/root/code/ComfyChat/server/source/knowledges/SaltAI-Web-Docs/md/a-person-mask-generator/licenses.md"
    # repo_name = "a-person-mask-generator"
    # add_first_title(md_path, repo_name)

    random_sample_question(alpaca_path="/root/code/ComfyChat/data/message_jsons/v1/alpaca_gpt4_data_zh_modification.json",
                           data_path="/root/code/ComfyChat/data/message_jsons/v2/community_zh.json",
                           can_questions_path="/root/code/ComfyChat/server/source/questions/zh/can_questions.json",
                           cannot_questions_path="/root/code/ComfyChat/server/source/questions/zh/cannot_questions.json")