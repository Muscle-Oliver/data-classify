# coding: UTF-8
from config import DefaultConfig
import time
import os
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
import pickle as pkl
import time
from utils import build_vocab, build_dataset, get_time_dif, BasicCollate
import random
import pandas as pd
import datasets
import transforms
from torch.utils.data import DataLoader
from clearml import Task
task = Task.get_task(project_name='HPO Auto-training', task_name='Datatype Classifier')

#声明argparse对象 可附加说明
parser = argparse.ArgumentParser(description='Chinese Text Classification')

parser.add_argument('--model', default='RNN',type=str, help='choose a model: RNN,RNN_FIXED')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')

args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

MAX_VOCAB_SIZE = 10000

if __name__ == '__main__':
    
    dataset = 'DC'  # 数据集
    if args.embedding == 'random': #如果embedding参数设置为random
        print('使用随机生成')
        embedding = 'random'
    else:
        print('使用字向量')
        embedding = 'embedding_SougouNews_char.npz'

    #生成字典和预训练矩阵
    data_dir = "./DC/data/train.txt"
    #vocab_dir = "./TC/data/vocab_word.pkl"
    vocab_dir = "./DC/data/vocab.pkl"
    #pretrain_dir = "./TC/data/sgns.sogou.word"
    pretrain_dir = "./source/sgns.sogou.char"
    #词/字嵌入矩阵存储路径
    filename_trimmed_dir = "./DC/data/embedding_SougouNews_char.npz"
    embed_size = 300

    if os.path.exists(filename_trimmed_dir):
        print('初始化的词/字嵌入矩阵已存在')
    else:
        if os.path.exists(vocab_dir): #如果有处理好的词/字典
            # print('加载已存在的字典')
            word_to_id = pkl.load(open(vocab_dir, 'rb'))
        else:  #构建词/字典（基于训练集）
            # print('没有存在的字典，现在生成并保存')
            # tokenizer = lambda x: list(jieba.cut(x))  # 以词为单位构建词表
            tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
            
            #构建词/字典
            word_to_id = build_vocab(data_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            #保存词/字典
            pkl.dump(word_to_id, open(vocab_dir, 'wb'))

        embeddings = np.random.rand(len(word_to_id), embed_size) #随机初始化词/字嵌入矩阵

        #读取预训练词/字向量
        f = open(pretrain_dir, "r", encoding='UTF-8')
        for i, line in enumerate(f.readlines()): #遍历每一行 格式：词/字 300个数字(均以空格分开)
            #用预训练词/字向量覆盖 随机初始化的词/字嵌入矩阵
            lin = line.strip().split(" ")
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        f.close()

        #保存初始化的词/字嵌入矩阵
        np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
        print('保存初始化的词/字嵌入矩阵 save to %s' % filename_trimmed_dir)        
    
    # #获取选择的模型名字
    model_name = args.model 

    x = import_module('models.' + model_name) #根据所选模型名字在models包下 获取相应模块(.py)
    # config = x.Config(dataset, embedding) #每一个模块(.py)中都有一个模型定义类 和与该模型相关的配置类(定义该模型的超参数) 初始化配置类的对象
    config = DefaultConfig()
    config.model_name = model_name
    print('----model name----', model_name)   
    set_seed(3)
    
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    tokenizer = lambda x: [y for y in x]
    labels2index = pkl.load(open(config.labels2index_path, 'rb'))
    print("labels2index::::::::::::::::", labels2index)
    task.upload_artifact(name='word2index', artifact_object=vocab)
    task.upload_artifact(name='labels2index', artifact_object=labels2index)
    task.upload_artifact(name='Inference info', artifact_object={"max_input_length": 50, "output_length": len(labels2index)})
    
    #直接沿用modify_master的自行构造dataset在transform的时候实现index化，会在每次用loader调用的时候非常慢，尤其在数据量很大的时候特别明显。
    # 因此将text和label 转换为index，在构造dataset之前。另外简单构造dataset，加入build_dataset函数
    
    preprocess_start = time.time()   
    
    #将TypeClassifier改写，在__init__的时候就将text和label转换成index，跟用build_dataset输出结果一样，这里这样写只是为了保证项目的规范性，用自定义的dataset类来构造dataset
    train_dataset = datasets.typeclassifier.TypeClassifier_v2(config.train_path,tokenizer,vocab,labels2index)
    print('building train_data',len(train_dataset))
    dev_dataset = datasets.typeclassifier.TypeClassifier_v2(config.dev_path,tokenizer,vocab,labels2index)
    print('building dev_data',len(dev_dataset))
    test_dataset = datasets.typeclassifier.TypeClassifier_v2(config.test_path,tokenizer,vocab,labels2index)
    print('building test_data',len(test_dataset))
    
    mycollate = BasicCollate(config.device, config.vocab_path)    
    train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=mycollate)
    dev_loader = DataLoader(dataset=dev_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=mycollate)
    test_loader = DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=mycollate)

    # # 构造模型对象
    config.n_vocab = len(vocab) #词典大小可能不确定，在运行时赋值
    model = x.Model(config.n_vocab, config).to(config.device) #构建模型对象 并to_device

    start_time = time.time()
    print("Training begin....")
    #训练、验证和测试
    train(config, model, train_loader, dev_loader, test_loader)

    time_dif = get_time_dif(start_time)
    print("Training is over, and time usage:", time_dif)
    
