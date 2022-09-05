# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
import time
from datetime import timedelta
import copy
import numpy as np
import math
import random
import torch.utils.data as torchdata

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def get_and_format_data(file_name):
    f = open(file_name,encoding='utf-8')
    lines = f.read().strip().split('\n')
    # random.shuffle(lines)
    input_list = [] 
    target_list = []   
    for num,line in enumerate(lines):
        line = line.split('@')
        #提取input数据
        input_list.append(line[0])
        target_list.append(line[1])
    f.close()    
    return input_list, target_list


def get_index(s, tokenizer, vocab):
    token = tokenizer(s) 
    seq_len = len(token) 
    words_line = []
    for word in token: #将词/字转换为索引，不在词/字典中的 用UNK对应的索引代替
        words_line.append(vocab.get(word, vocab.get(UNK)))
    return words_line


def build_vocab(data_path, tokenizer, max_size, min_freq):
    data, _ = get_and_format_data(data_path)
    #词/字典
    vocab_dic = {}
    for item in data:
        for word in tokenizer(item): #分词 or 分字
            vocab_dic[word] = vocab_dic.get(word, 0) + 1 #构建词或字到频数的映射 统计词频/字频

    #根据 min_freq过滤低频词，并按频数从大到小排序，然后取前max_size个单词
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    #构建词或字到索引的映射 从0开始
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    #添加未知符和填充符的映射
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    print('vocab size',len(vocab_dic))
    return vocab_dic

 
class BasicCollate(object):
    def __init__(self, device, vocab_path):
        self.device = device
        self.vocab_path = vocab_path

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pkl.load(f)

    def _collate(self, batch):
        """
        Realize pad and tensorization, this operations should be done during the step of generating batch data namely 'Dataloader'.
                
        """  
        # print('batch',batch)
        #vocab = pkl.load(open(self.vocab_path, 'rb'))

        xs_list = [v[0] for v in batch]
        ys = torch.LongTensor([v[1] for v in batch])#.to(self.device)
        # 获得每个样本的序列长度
        seq_lengths = torch.LongTensor([v for v in map(len, xs_list)])#.to(self.device)
        max_len = max([len(v) for v in xs_list])
        # 每个样本都padding到当前batch的最大长度
        for i in xs_list:
            if len(i) < max_len:
                i.extend([self.vocab.get(PAD)]*(max_len - len(i)))
        xs = torch.LongTensor(xs_list)#.to(self.device)
        # print('xs',xs)
        # print('seq_lengths',seq_lengths)
        # print('ys',ys)
        del xs_list
        del batch
        return xs, seq_lengths, ys

    def __call__(self, batch):
        return self._collate(batch)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def text_to_index(text_list, tokenizer, vocab):
    index_list = []
    for i in text_list:
        index_list.append(get_index(i, tokenizer, vocab))
    return index_list

def label_to_index(label_list, labels2index):
    index_list = []
    for i in label_list:
        index_list.append(labels2index[i])
    return index_list

'''
def build_dataset(data_path, tokenizer, vocab, labels2index):
    X, y = get_and_format_data(data_path)
    X = text_to_index(X, tokenizer, vocab)
    y = label_to_index(y, labels2index)

    zipped=zip(X,y)
    return list(zipped)
'''
def build_dataset(data_path, tokenizer, vocab, labels2index):
    return TextDataset(data_path, tokenizer, vocab, labels2index)

import copy

class TextDataset(torchdata.Dataset):
    def __init__(self, data_path, tokenizer, vocab, labels2index):

        X, y = get_and_format_data(data_path)
        X = text_to_index(X, tokenizer, vocab)
        y = label_to_index(y, labels2index)

        zipped=zip(X,y)
        self.data = list(zipped)

    def __getitem__(self, index):
        
        #这里直接返回self.data[index]会造成内存泄漏 
        return copy.deepcopy(self.data[index])

    def __len__(self):
        return len(self.data)
 
        

def index_to_text(index_list):
    text_list = []
    with open('./DC/data/index2vocb.pkl', 'rb') as f:
        index2vocab = pkl.load(f)
    for i in index_list:
        text_list.append(index2vocab[i])
    text = ''.join(text_list).replace('<PAD>','')
    return text

def index_to_label(index):
    with open('./DC/data/index2labels.pkl', 'rb') as f:
        index2labels = pkl.load(f)
    return index2labels[index] 
