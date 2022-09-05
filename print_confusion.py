# coding: UTF-8
from config import opt
import time
import os
import torch
import numpy as np
from train_eval import test, evaluate
from importlib import import_module
import argparse
import pickle as pkl
import time
from utils import build_vocab, build_dataset, get_time_dif, BasicCollate
import random
import datasets
import transforms
from torch.utils.data import DataLoader


if __name__ == '__main__':

    # #生成字典和预训练矩阵
    # data_dir = "./DC/data/train.txt"
    # #vocab_dir = "./TC/data/vocab_word.pkl"
    # vocab_dir = "./DC/data/vocab.pkl"
    # #pretrain_dir = "./TC/data/sgns.sogou.word"
    # pretrain_dir = "./source/sgns.sogou.char"
    # #词/字嵌入矩阵存储路径
    # filename_trimmed_dir = "./DC/data/embedding_SougouNews_char.npz"
    # embed_size = 300

    # if os.path.exists(filename_trimmed_dir):
    #     print('初始化的词/字嵌入矩阵已存在')
    # else:
    #     if os.path.exists(vocab_dir): #如果有处理好的词/字典
    #         # print('加载已存在的字典')
    #         word_to_id = pkl.load(open(vocab_dir, 'rb'))
    #     else:  #构建词/字典（基于训练集）
    #         # print('没有存在的字典，现在生成并保存')
    #         # tokenizer = lambda x: list(jieba.cut(x))  # 以词为单位构建词表
    #         tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
            
    #         #构建词/字典
    #         word_to_id = build_vocab(data_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #         #保存词/字典
    #         pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    #     embeddings = np.random.rand(len(word_to_id), embed_size) #随机初始化词/字嵌入矩阵

    #     #读取预训练词/字向量
    #     f = open(pretrain_dir, "r", encoding='UTF-8')
    #     for i, line in enumerate(f.readlines()): #遍历每一行 格式：词/字 300个数字(均以空格分开)
    #         #用预训练词/字向量覆盖 随机初始化的词/字嵌入矩阵
    #         lin = line.strip().split(" ")
    #         if lin[0] in word_to_id:
    #             idx = word_to_id[lin[0]]
    #             emb = [float(x) for x in lin[1:301]]
    #             embeddings[idx] = np.asarray(emb, dtype='float32')
    #     f.close()

    #     #保存初始化的词/字嵌入矩阵
    #     np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    #     print('保存初始化的词/字嵌入矩阵 save to %s' % filename_trimmed_dir)        
    
    # #获取选择的模型名字
    model_name = 'RNN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    x = import_module('models.' + model_name) #根据所选模型名字在models包下 获取相应模块(.py)
    # config = x.Config(dataset, embedding) #每一个模块(.py)中都有一个模型定义类 和与该模型相关的配置类(定义该模型的超参数) 初始化配置类的对象
    config = opt
    config.model_name = model_name
    print('----model name----', model_name)   
    
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    tokenizer = lambda x: [y for y in x]
    labels2index = pkl.load(open('./DC/data/labels2index.pkl', 'rb'))
    print("labels2index", labels2index)
    preprocess_start = time.time()   
    train_dataset = build_dataset(config.train_path, tokenizer, vocab, labels2index)
    print('building train_data',len(train_dataset))
    print(train_dataset[:5])
    dev_dataset = build_dataset(config.dev_path, tokenizer, vocab, labels2index)
    print('building dev_data',len(dev_dataset))
    test_dataset = build_dataset(config.test_path, tokenizer, vocab, labels2index)
    print('building test_data',len(test_dataset))
    print('build dataset time usage..........',time.time() - preprocess_start)
 
    mycollate = BasicCollate(config.device, config.vocab_path)    
    train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=mycollate)
    dev_loader = DataLoader(dataset=dev_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=mycollate)
    test_loader = DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=mycollate)

    # # 构造模型对象
    config.n_vocab = len(vocab) #词典大小可能不确定，在运行时赋值
    model = x.Model(config.n_vocab, config).to(config.device) #构建模型对象 并to_device
    
    #加载训练好的模型参数
    model.load_state_dict(torch.load('./DC/saved_dict/'+ model_name + '_5.pth'))

    test(model, train_loader)
    print('=======================')
    test(model, test_loader)

