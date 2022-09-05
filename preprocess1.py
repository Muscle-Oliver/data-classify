# coding: UTF-8
from config import opt
import copy
from importlib import import_module
import glob
import os
import io
import math
import numpy as np
import random
import json
import pickle as pkl
import collections
import torch
import sys
import re
import random
from clearml import Dataset
sys.path.append('../')

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def random_cut(line,len_min,len_max):
    line_list = line.split(';')
    count = 0
    while count<50:
        count += 1
        p = random.randint(0,len(line_list)-1)
        l = random.randint(len_min,len_max)
        if p+l >= len(line_list)-1:
            continue
        tem_line = line_list[p:p+l]
        break
    if count==50:
        if len(line_list) > 1:
            tem_line=line_list[0:2]
        else:
            tem_line=line_list
    result = ''.join(tem_line)

    return result


def get_data(data_path,background_data_path,save_folder):
        all_filenames = glob.glob(data_path)
        # Build the category_lines dictionary, a list of names per classicification
        category_lines = {}
        all_categories = []
        # Read a file and split into lines
        def readLines(filename):
            f = open(filename,encoding='utf-8')
            # f = open(filename)
            lines = f.read().replace("@", "").strip().split('\n')
            f.close()
            return lines

        for filename in all_filenames:
            category = filename.split('/')[-1].split('.')[0].split('\\')[-1]
            # category = filename.split('/')[-1].split('.')[0]
            all_categories.append(category)
            lines = readLines(filename)
            lines_list = []
            for line in lines:
                if line != '':
                    lines_list.append(line)
                category_lines[category] = lines_list
            
            if category != '毕业学校':
                category_lines[category] = list(set(category_lines[category]))
            print(category)
            print(len(category_lines[category]))

        print("categories include:")
        print(all_categories)
        # n_categories = len(all_categories)
        data = []

        aa = 0
        for category in all_categories:
            for item in category_lines[category]:
                # punc = '(|)|（|）|;'
                # item = re.sub("[%s]+" %punc, "", item)
                item = item.strip()
                if line is None:
                    print("11111111111111")
                if line is np.nan:
                    print("11111111111111")
                if line == '':
                    print("11111111111111")
                data.append([item, category])
                aa += 1
                # print(aa)

        print('there are %d intries....' % (len(data)))

        background_data = []
        with open('./source/background_data_2.txt',encoding = 'utf-8', mode='r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                # print(line)
                background_data.append([line, '背景'])
        print('background_data has :', len(background_data))
        print(background_data[:5])
        category_lines["背景"] = [x[0] for x in background_data]

        data = data + background_data

        all_categories.append("背景")
        print("种类总共有：")
        print(all_categories)

        def label2index(labels):
            labels2index = dict(zip(labels, list(range(len(labels)))))
            index2labels = dict(zip(list(range(len(labels))),labels))
            print(labels2index)
            print(index2labels)
            with open(save_folder + '/index2labels.pkl','wb') as f:
                pkl.dump(index2labels,f)
            with open(save_folder + '/labels2index.pkl','wb') as f:
                pkl.dump(labels2index,f)
            with open(save_folder + '/labels.pkl','wb') as f:
                pkl.dump(labels,f)
            return labels2index

        labels2index=label2index(all_categories)

        return data, category_lines

def clean_background(data, background_data):
    background_data_copy = copy.deepcopy(background_data)

    remove_list = ['公司','厂','店','公园','医院','卫生院','大学','学院','小学','中学','学校']
    for i in background_data_copy:
        for j in remove_list:
            if j in i[0]:
                if i in background_data:
                    background_data.remove(i)
                    print('remove field data!!!!!', i)

    print('background_data', len(background_data))
    print('background_data_copy', len(background_data_copy))

    for item in data:
        for i in background_data_copy:        
            if i[0] == item[0]:
                if i in background_data:
                    background_data.remove(i)
                    print('remove same data!!!!!', i)

    print('background_data', len(background_data))
    print('background_data_copy', len(background_data_copy))

    #using existed model to clean background
    print('using model to select unproper data...')
    model_name = 'RNN_p'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #加载之前预处理过程 保存的词到索引的映射字典
    with open('vocab.pkl', 'rb') as f:
        word2index = pkl.load(f)

    # 加载之前预处理过程 保存的索引到类别标签的映射字典
    with open('index2labels.pkl', 'rb') as f:
        index2labels = pkl.load(f)

    x = import_module('models.' + model_name)
    config = opt
    model = x.Model(len(index2labels), config).to(config.device) #构建模型对象 并to_device

    model.load_state_dict(torch.load(model_name + '.pth', map_location=device))

    for i in background_data_copy:
        text = i[0]
        sentence = torch.tensor([word2index.get(word,word2index.get(UNK)) for word in text],device=device)
        with torch.no_grad():
            model.eval()
            x_test = sentence.view((1,-1))
            length = len(sentence)
            seq = torch.tensor([length])
            y_hat = model(x_test,seq)
            label = torch.argmax(y_hat,dim=1)            
            if  index2labels[label.item()] == '姓名':
                if i in background_data:
                    background_data.remove(i)
                    print('@@@@@', text, '@'+index2labels[label.item()])

    print('background_data', len(background_data))
    print('background_data_copy', len(background_data_copy))

    return background_data

#split to train,val,test dataset 0.7:0.05:0.25
def split_data(data,save_folder):
    train_data = []
    val_data = []
    test_data = []

    np.random.shuffle(data)
    data_size = len(data)
    train_size = int(data_size*0.7)
    val_size = int(data_size*0.05)
    train_data = data[:train_size]
    #with open('./data/' + save_folder +'/labels2index.json') as f:
    with open(save_folder +'/labels2index.pkl','rb') as f:
        labels2index = pkl.load(f)

    np.random.shuffle(train_data) #打乱
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    print('train_original',len(train_data))

    return train_data,val_data,test_data

#split to train,val,test dataset 0.7:0.05:0.25
def split_duplicate(data, category_lines, save_folder, source_data_folder):
    print('###########',len(data))
    print(data[:10])
    train_data = []
    test_data = []

    category_list = list(category_lines.keys())
    category_length = [len(category_lines[x]) for x in category_list]
    
    x = 0
    while x < len(category_list):
        if len(category_lines[category_list[x]]) == 0:
            category_lines.pop(x)
            category_list.pop(x)
        else:
            x += 1
    
    max_length = max(category_length)
    duplicate_list = [math.floor(max_length / x) for x in category_length]
    
    for item in zip(category_list,duplicate_list):
        category = item[0]
        print('category',category)
        duplicate = item[1]
        print('duplicate',duplicate)
        temp_list = []
        for i in data:
            if i[1] == category:
                temp_list.append(i)
        np.random.shuffle(temp_list)
        temp_size = len(temp_list)
        print('temp_size',temp_size)
        
        if category != '公司名称' or category == '背景' or category == '毕业学校' or category == '姓名':
            temp_train = temp_list*duplicate
            temp_test = temp_list[int(temp_size*0.95):]*duplicate
        else:
            temp_train = temp_list[:int(temp_size*0.95)]*duplicate
            temp_test = temp_list[int(temp_size*0.95):]*duplicate
        
        np.random.shuffle(temp_train)
        np.random.shuffle(temp_test)
        print(len(temp_train))
        # print(len(temp_val))
        print(len(temp_test))
        train_data = train_data + temp_train
        # val_data = val_data + temp_val
        test_data = test_data + temp_test
        np.random.shuffle(train_data)
        # np.random.shuffle(val_data)
        np.random.shuffle(test_data)

    # return train_data,val_data,test_data
    return train_data,test_data


def add_self_define(dataset_selfdefine, train_data, duplicate_num):
    l=[]
    with open(dataset_selfdefine,encoding = 'utf-8', mode='r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            line = line.split('@')
            l.append([line[0], line[1]])

    l = l * duplicate_num
    print('add self define',len(l),l[:5])
    #shuffle list_new
    np.random.shuffle(l)
    train_data = train_data + l
    np.random.shuffle(train_data)
    return train_data


if __name__ == '__main__':
    source_data_folder = './source/target'
    
    ### load clearml dataset
    try:
        dataset_name = os.getenv("DATASET_NAME", "Dataset Demo")
        dataset_project = os.getenv("DATASET_PROJECT", "Dataset")
        
        dataset_path = Dataset.get(
            dataset_name=dataset_name, 
            dataset_project=dataset_project
        ).get_local_copy()
        json_path = glob.glob(f"{dataset_path}/*")
        for p in json_path:
            with open(p, encoding='utf-8', mode='r') as f:
                json_content = json.loads(f.read())
            data = json_content['task']['data']
            label = json_content['result'][0]['value']['taxonomy'][0][0]
            with open(f"{source_data_folder}/{label}.txt", encoding='utf-8', mode='w') as f:
                f.write("\n".join(data.values()))
    except:
        print(f'Dataset {dataset_name} not found on server, use local dataset instead.')
    ### load clearml dataset
    
    background_data_path = './source/background_shuffle_100000.txt'
    save_folder = './DC/data'
    vocab_dir = save_folder + '/word2index.pkl'

    data, category_lines = get_data(source_data_folder + '/*.txt', background_data_path, save_folder)
    print(len(data))
    # train_data, val_data, test_data = split_data(data,save_folder)
    train_data, test_data = split_duplicate(data, category_lines, save_folder, source_data_folder)

    print('===============train data is like=============')
    print(train_data[:5])

    train_data = add_self_define('./source/self_define1.txt', train_data, 2)
    print('train_data has been added self-defined data and sum to :%d...' % len(train_data))
    
    print('train',len(train_data))
    # print('val',len(val_data))
    print('test',len(test_data))

    with open(save_folder + '/train.txt',encoding = 'utf-8', mode='w') as f:
        for line in train_data:
            f.write(line[0]+'@'+str(line[1])+'\n')
    
    with open(save_folder + '/test.txt',encoding = 'utf-8', mode='w') as f:
        for line in test_data:
            f.write(line[0]+'@'+str(line[1])+'\n')
    print("-----------------test dataset-----------------")
    print(test_data[:20])
