import torch
import torch.utils.data as data
import numpy as np
from utils import get_and_format_data, text_to_index, label_to_index
import copy

class TypeClassifier(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root= root
        self.transform = transform
        self.target_transform = target_transform

        self.datas, self.targets = get_and_format_data(self.root)

        # print('datas',len(self.datas),self.datas[-3:])
        # print('targets',len(self.targets),self.targets[-3:])

    def __getitem__(self, index):
        data, target = self.datas[index], self.targets[index]
        # print('index',index)
        # print('__getitem__:')
        # print(data)
        # print(target)
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #这里直接返回会造成内存泄漏  
        return data, target

    def __len__(self):
        return len(self.datas)


class TypeClassifier_v2(data.Dataset):

    def __init__(self, root, tokenizer, vocab, labels2index, transform=None, target_transform=None):
        # self.root= root
        # self.transform = transform
        # self.target_transform = target_transform
        # self.datas, self.targets = get_and_format_data(self.root)
        X, y = get_and_format_data(root)
        X = text_to_index(X, tokenizer, vocab)
        y = label_to_index(y, labels2index)

        zipped=zip(X,y)
        self.data = list(zipped)

    def __getitem__(self, index):
        # data, target = self.datas[index], self.targets[index]
        # if self.transform is not None:
        #     data = self.transform(data)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # return data, target
        
        #这里直接返回self.data[index]会造成内存泄漏  
        return copy.deepcopy(self.data[index])
        #return self.data[index]

    def __len__(self):
        return len(self.data)

class TypeClassifier_v3(data.Dataset):

    def __init__(self, root, tokenizer, vocab, labels2index, transform=None, target_transform=None):
        # self.root= root
        # self.transform = transform
        # self.target_transform = target_transform
        # self.datas, self.targets = get_and_format_data(self.root)
        X, y = get_and_format_data(root)
        X = text_to_index(X, tokenizer, vocab)
        y = label_to_index(y, labels2index)
    
        zipped=zip(X,y)
        self.data = list(zipped)
        self.transform = transform
    def __getitem__(self, index):
        #transform 使用NameTransform，姓名类的数据，进行NameTransform操作，其它类别数据不做改动
        # if self.transform is not None:
        if self.transform is not None:
            self.data[index] = self.transform(self.data[index])
        
        #这里直接返回self.data[index]会造成内存泄漏  
        return copy.deepcopy(self.data[index])
        #return self.data[index]

    def __len__(self):
        return len(self.data)


class Pre_TableGrade(data.Dataset):

    def __init__(self, datas, targets, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.datas, self.targets = datas, targets

        # print('datas',len(self.datas),self.datas[-3:])
        # print('targets',len(self.targets),self.targets[-3:])

    def __getitem__(self, index):
        data, target = self.datas[index], self.targets[index]
        # print('index',index)
        # print('__getitem__:')
        # print(data)
        # print(target)
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(data)
        # print(target)
        return data, target

    def __len__(self):
        return len(self.datas)




