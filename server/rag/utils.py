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
import hashlib
import datetime
import logging
from logging import FileHandler, StreamHandler, Formatter

import fitz
import textract
import pandas as pd
from bs4 import BeautifulSoup


def create_logger(name):
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 输出到控制台
    c_handler = StreamHandler(sys.stdout)
    c_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(c_handler)
    
    # 日志保存在app.log中，设置日志等级和格式
    time_flag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    f_handler = FileHandler(f'./logs/{name}_{time_flag}.log', encoding='utf-8')
    f_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(f_handler)
    
    return logger


logger = create_logger("embedding_extract")


class FileName:
    """Record file original name, state and copied filepath with text format."""

    def __init__(self, root: str, filename: str, _type: str):
        self.root = root
        self.prefix = filename.replace('/', '_')
        self.basename = os.path.basename(filename)
        self.origin = os.path.join(root, filename)
        self.copypath = ''
        self._type = _type  # 文件类型
        self.state = True  # 是否抽取成功
        self.reason = ''

    def __str__(self):
        return '{},{},{},{}\n'.format(self.basename, self.copypath, self.state,
                                      self.reason)


class FileOperation:
    """Encapsulate all file reading operations."""

    def __init__(self):
        self.image_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
        self.md_suffix = '.md'
        self.text_suffix = ['.txt', '.text']
        self.excel_suffix = ['.xlsx', '.xls', '.csv']
        self.pdf_suffix = '.pdf'
        self.ppt_suffix = '.pptx'
        self.html_suffix = ['.html', '.htm', '.shtml', '.xhtml']
        self.word_suffix = ['.docx', '.doc']
        # 除图片后缀的所有文本文件后缀
        self.normal_suffix = [self.md_suffix
                              ] + self.text_suffix + self.excel_suffix + [
                                  self.pdf_suffix
                              ] + self.word_suffix + [self.ppt_suffix
                                                      ] + self.html_suffix

    # 返回文件所属类型
    def get_type(self, filepath: str):
        filepath = filepath.lower()
        if filepath.endswith(self.pdf_suffix):
            return 'pdf'

        if filepath.endswith(self.md_suffix):
            return 'md'

        if filepath.endswith(self.ppt_suffix):
            return 'ppt'

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
    def md5(self, filepath: str):
        hash_object = hashlib.sha256()
        with open(filepath, 'rb') as file:
            chunk_size = 8192  # 8 KB
            while chunk := file.read(chunk_size):
                hash_object.update(chunk)  # 将每个chunk更新到hash_object中，防止文件过大导致内存不足

        return hash_object.hexdigest()[0:8]  # 最终返回整个文件sha256值的前八位

    # 对所有文件的处理情况进行记录
    def summarize(self, files: list):
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
    def scan_dir(self, repo_dir: str):
        files = []
        for root, _, filenames in os.walk(repo_dir):
            for filename in filenames:
                _type = self.get_type(filename)
                if _type is not None:
                    files.append(
                        FileName(root=root, filename=filename, _type=_type))
        return files

    def read_pdf(self, filepath: str):
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

    def read_excel(self, filepath: str):
        table = None
        if filepath.endswith('.csv'):
            table = pd.read_csv(filepath)
        else:
            table = pd.read_excel(filepath)
        if table is None:
            return ''
        json_text = table.dropna(axis=1).to_json(force_ascii=False)
        return json_text

    def read(self, filepath: str):
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

    def __init__(self, log_file_path):
        """Initialize the QueryTracker with the path of the log file."""
        self.log_file_path = log_file_path
        self.log_list = []

    def log(self, key, value=''):
        """Log a query.

        Args:
            key (str): The key associated with the query.
            value (str): The value or result associated with the query.
        """
        self.log_list.append((key, value))

    def __del__(self):
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
    """Print histogram log string for values"""
    values.sort()
    _len = len(values)
    if _len < 1:
        return ''
    
    median = values[round((_len - 1) / 2)]
    _sum = 0
    min_val = min(values)
    max_val = max(values)
    range_width = round(0.1 * (max_val - min_val))
    if range_width == 0:
        logger.info("all input length = {}".format(min_val))
    ranges = [(i * range_width, (i + 1) * range_width) for i in range((max_val // range_width) + 1)]

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

    log_str = 'count {}, avg {}, median {}\n'.format(len(values), round(_sum / len(values), 2), median)
    for i, (start, end) in enumerate(ranges):
        log_str += f'{start}-{end}  {range_percentages[i]:.2f}%'
        log_str += '\n'
    return log_str


if __name__ == "__main__":
    file = r"D:\git_github\self\ComfyChat\data\custom_nodes_mds\a-person-mask-generator\README.md"
    fo = FileOperation()
    chars = fo.md5(file)
    print(chars)