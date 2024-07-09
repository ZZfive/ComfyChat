#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   retriever.py
@Time    :   2024/06/21 21:48:51
@Author  :   zzfive 
@Desc    :   None
'''


"""extract feature and search with user query."""
import os
import time
from typing import Tuple

import numpy as np
import pytoml
from BCEmbedding.tools.langchain import BCERerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores.faiss import FAISS as Vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from sklearn.metrics import precision_recall_curve

from util import FileOperation, QueryTracker, create_logger

logger = create_logger("retriever")


class Retriever:
    """Tokenize and extract features from the project's documents, for use in the reject pipeline and response pipeline."""
    """此类中的reject将从拒绝回答转变为RAG未检索到相关内容，直接使用llms回答的标志"""

    def __init__(self, embeddings, reranker, work_dir: str, reject_throttle: float) -> None:
        """Init with model device type and config."""
        self.reject_throttle = reject_throttle
        self.rejecter = None
        self.retriever = None
        self.compression_retriever = None

        if not os.path.exists(work_dir):
            logger.warning('!!!warning, workdir not exist.!!!')
            return

        rejection_path = os.path.join(work_dir, 'db_reject')
        retriever_path = os.path.join(work_dir, 'db_response')

        if os.path.exists(rejection_path):  # 加载判断“当前RAG能否回答”一个query的向量存储
            self.rejecter = Vectorstore.load_local(
                rejection_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True)
        
        if os.path.exists(retriever_path):  # 加载用于RAG回答query的向量存储
            self.retriever = Vectorstore.load_local(
                retriever_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
                    search_type='similarity',
                    search_kwargs={
                        'score_threshold': 0.15,
                        'k': 30
                    })
            self.compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=self.retriever)
        
        if self.rejecter is None:
            logger.warning('rejecter is None')
        if self.retriever is None:
            logger.warning('retriever is None')

    def is_relative(self, question, k=30, disable_throttle=False) -> Tuple[bool, list]:
        """If no search results below the threshold can be found from the database, reject this query."""
        """如果从数据库中找不到低于阈值的搜索结果，则拒绝此查询。"""

        if self.rejecter is None:
            return False, []

        if disable_throttle:  # 不用卡阈值，直接搜索，但只取一个
            # for searching throttle during update sample
            docs_with_score = self.rejecter.similarity_search_with_relevance_scores(
                question, k=1)
            if len(docs_with_score) < 1:
                return False, docs_with_score
            return True, docs_with_score
        else:
            # for retrieve result
            # if no chunk passed the throttle, give the max
            docs_with_score = self.rejecter.similarity_search_with_relevance_scores(
                question, k=k)
            ret = []
            max_score = -1
            top1 = None
            for (doc, score) in docs_with_score:
                if score >= self.reject_throttle:
                    ret.append(doc)
                if score > max_score:
                    max_score = score
                    top1 = (doc, score)
            relative = True if len(ret) > 0 else False
            return relative, [top1]  # 返回top_k中score最大的

    # 基于设置的可以回答问题列表和不能回答问题列表，更新RAG不能回答问题的阈值
    def update_throttle(self,
                        config_path: str = 'config.ini',
                        can_questions=[],
                        cannot_questions=[]) -> None:
        """Update reject throttle based on positive and negative examples."""

        if len(can_questions) == 0 or len(cannot_questions) == 0:
            raise Exception('good and bad question examples cat not be empty.')
        questions = can_questions + cannot_questions
        predictions = []
        for question in questions:
            self.reject_throttle = -1
            _, docs = self.is_relative(question=question, disable_throttle=True)
            score = docs[0][1]
            predictions.append(max(0, score))

        labels = [1 for _ in range(len(can_questions))
                  ] + [0 for _ in range(len(cannot_questions))]
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions)

        # get the best index for sum(precision, recall)
        sum_precision_recall = precision[:-1] + recall[:-1]
        index_max = np.argmax(sum_precision_recall)
        optimal_threshold = max(thresholds[index_max], 0.0)  # 根据预测结果确定最优的阈值

        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)
        config['feature_store']['reject_throttle'] = float(optimal_threshold)
        with open(config_path, 'w', encoding='utf8') as f:
            pytoml.dump(config, f)
        logger.info(
            f'The optimal threshold is: {optimal_threshold}, saved it to {config_path}'  # noqa E501
        )

    # 从向量数据中查询与传入question相关的文本信息
    def query(self, question: str, context_max_length: int = 16000, tracker: QueryTracker = None) -> Tuple[str, str, list]:
        """Processes a query and returns the best match from the vector store database. If the question is rejected, returns None.

        Args:
            question (str): The question asked by the user.
            context_max_length: The maximum length of the context to return.
        Returns:
            str: The best matching chunk, or None.
            str: The best matching text, or None
        """
        if question is None or len(question) < 1:
            return None, None, []

        if len(question) > 512:
            logger.warning('input too long, truncate to 512')
            question = question[0:512]

        chunks = []
        context = ''
        references = []

        relative, docs = self.is_relative(question=question)
        logger.debug('retriever.docs {}'.format(docs))
        if not relative:  # 不相关，直接返回
            if len(docs) > 0:
                references.append(docs[0][0].metadata['source'])
            return None, None, references

        docs = self.compression_retriever.get_relevant_documents(question)  # 基于向量数据检索到的文件
        if tracker is not None:
            tracker.log('retrieve', [doc.metadata['source'] for doc in docs])  # 记录检索到的文件

        # add file text to context, until exceed `context_max_length`

        file_opr = FileOperation()
        for idx, doc in enumerate(docs):
            chunk = doc.page_content
            chunks.append(chunk)

            if 'read' not in doc.metadata:
                logger.error('If you are using the version before 20240319, please rerun `python3 -m huixiangdou.service.feature_store`')
                raise Exception('huixiangdou version mismatch')
            
            file_text, error = file_opr.read(doc.metadata['read'])  # 读取文件原始文本
            if error is not None:
                # read file failed, skip
                continue

            source = doc.metadata['source']  # 文件路径
            logger.info('target {} file length {}'.format(source, len(file_text)))
            if len(file_text) + len(context) > context_max_length:
                if source in references:
                    continue  # 没有break的原因是剩下的长度虽然不能放下当前的chunk，但可能装下后续的chunk
                references.append(source)
                # add and break
                add_len = context_max_length - len(context)
                if add_len <= 0:
                    break
                chunk_index = file_text.find(chunk)
                if chunk_index == -1: # chunk not in file_text
                    context += chunk
                    context += '\n'
                    context += file_text[0:add_len - len(chunk) - 1]
                else:
                    start_index = max(0, chunk_index - (add_len - len(chunk)))
                    context += file_text[start_index: start_index + add_len]
                break

            if source not in references:
                context += file_text
                context += '\n'
                references.append(source)

        context = context[0: context_max_length]
        logger.debug('query:{} top1 file:{}'.format(question, references[0]))
        return '\n'.join(chunks), context, [os.path.basename(r) for r in references]


class CacheRetriever:

    def __init__(self, config_path: str, max_len: int = 4) -> None:
        self.cache = dict()
        self.max_len = max_len
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)['feature_store']
            embedding_model_path = config['embedding_model_path']
            reranker_model_path = config['reranker_model_path']

        # load text2vec and rerank model
        logger.info('loading test2vec and rerank models')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={
                'batch_size': 1,
                'normalize_embeddings': True
            })
        self.embeddings.client = self.embeddings.client.half()
        reranker_args = {
            'model': reranker_model_path,
            'top_n': 7,
            'device': 'cuda',
            'use_fp16': True
        }
        self.reranker = BCERerank(**reranker_args)

    def get(self,
            fs_id: str = 'default',
            config_path='config.ini',
            work_dir='workdir') -> Retriever:
        if fs_id in self.cache:
            self.cache[fs_id]['time'] = time.time()  # 更新时间
            return self.cache[fs_id]['retriever']  # 返回默认retriever

        with open(config_path, encoding='utf-8') as f:
            reject_throttle = pytoml.load(f)['feature_store']['reject_throttle']  # 加载据答阈值

        if len(self.cache) >= self.max_len:  # cache中的retriever超过个数，删除最久远的
            # drop the oldest one
            del_key = None
            min_time = time.time()
            for key, value in self.cache.items():
                cur_time = value['time']
                if cur_time < min_time:
                    min_time = cur_time
                    del_key = key

            if del_key is not None:
                del_value = self.cache[del_key]
                self.cache.pop(del_key)
                del del_value['retriever']

        retriever = Retriever(embeddings=self.embeddings,
                              reranker=self.reranker,
                              work_dir=work_dir,
                              reject_throttle=reject_throttle)  # 初始化新的retriever实例
        self.cache[fs_id] = {'retriever': retriever, 'time': time.time()}
        if retriever.rejecter is None:
            logger.warning('retriever.rejecter is None, check workdir')
        return retriever

    # 删除指定retriever
    def pop(self, fs_id: str) -> None:
        if fs_id not in self.cache:
            return
        del_value = self.cache[fs_id]
        self.cache.pop(fs_id)
        # manually free memory
        del del_value