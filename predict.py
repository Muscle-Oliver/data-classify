# coding: UTF-8
from config import DefaultConfig
from importlib import import_module
import torch
import pickle as pkl
import math
import random
import os
import time
import sys
from glob import glob

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
 
def get_data_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs)
    indices = list(range(rows))
    rest_number = batch_size
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:]  # 不循环移位，以便产生下一个batch
        batch_data = find_list(batch_indices, inputs)
        yield batch_data
 
def find_list(indices, data):
    out = []
    for i in indices:
        out.append(data[i])
    return out
 
def get_next_batch(batch):
    return batch.__next__()

def get_best_pth_path(pth_root="./nni/checkpoints"):
    acc_log_files = glob(os.path.abspath(f"{pth_root}/*.log"))
    acc_log_files.sort()
    assert len(acc_log_files) > 0
    trial_acc = []
    for path in acc_log_files:
        with open(path, "r") as fo:
            trial_acc.append(float(fo.read()))
    best_trial_id = trial_acc.index(max(trial_acc))
    best_pth_path = glob(os.path.abspath(f"{pth_root}/*{best_trial_id}.pth"))[0]
    return best_pth_path

if __name__ == '__main__':
    model_name = 'RNN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print("使用设备:", device)
    # text = '回龙镇' #待分类的单条文本
    text = None
    if sys.argv[1] != None:
        text = sys.argv[1]
    #prediction_path = './source/target/行政机关.txt' #待分类的文本文件
    prediction_path = None
    predict_batchsize = 100000000
    predict_batchsize = 256

    #加载之前预处理过程 保存的词到索引的映射字典
    with open('./DC/data/vocab.pkl', 'rb') as f:
        word2index = pkl.load(f)

    # 加载之前预处理过程 保存的索引到类别标签的映射字典
    with open('./DC/data/index2labels.pkl', 'rb') as f:
        index2labels = pkl.load(f)
    print("index2labels_dict:", index2labels)

    x = import_module('models.' + model_name)
    config = DefaultConfig()
    #model1 = x.Model(len(index2labels), config).to(config.device) #构建模型对象 并to_device
    #model2 = x.Model(len(index2labels), config).to(config.device) #构建模型对象 并to_device
    
    model = x.Model(len(index2labels), config).to(config.device)

    #加载训练好的模型参数
    # if device.type=='cpu': #GPU训练 CPU预测 加载参数时需要对参数进行映射
    #     model.load_map('./DC/saved_dict/'+ model_name + '_best.pth', device)
    # else:
    #     model.load('./DC/saved_dict/' + model_name + '_best.pth')
    
    #model1.load_state_dict(torch.load('./DC/saved_dict/'+ model_name + '_5.pth'))
    #model2.load_state_dict(torch.load('./DC/saved_dict/'+ model_name + '_best.pth'))
    
    best_pth_path = get_best_pth_path(pth_root="./nni/checkpoints")
    model.load_state_dict(torch.load(best_pth_path, map_location=device))
    
    #print("加载的训练模型参数用的是: %s" % list(model.parameters())[0].device)

    def pad(x,max_len):
        return x[:max_len] if len(x) > max_len else x + [word2index.get(PAD)] * (max_len - len(x))

    #预测文件
    if prediction_path != None:
        #一次性读取预测文本文件，形成list，后面批量预测并处理生成结果
        text_list = open(prediction_path).read().strip().split('\n')
        print('predict text includes :', len(text_list), text_list[:2])

        #批量处理文本
        if (len(text_list) % predict_batchsize) != 0:
            iter = len(text_list) // predict_batchsize + 1  # 迭代完全部数据，每次输出一个batch个
        else:
            iter = len(text_list) // predict_batchsize
        batch = get_data_batch(text_list, predict_batchsize, shuffle=False)

        count = 0
        save_list = []
        for i in range(iter):
            print("第%d批次..." % (i+1))
            batch_text = get_next_batch(batch)
            #batch_text = batch_text*5000
            #print("###################################")
            #print(len(batch_text))
            #print(batch_text[0:5])
            batch_text=[str(i).strip() for i in batch_text if str(i)!="nan"]
            #print(batch_text)
            device = list(model.parameters())[0].device

            #获取batch对应的seq，以便在pad之后要用变长序列的模型处理
            seq_len = torch.LongTensor([v for v in map(len, batch_text)])
            #获取该batch中最长文本的长度，以便将整个batch都pad成这个长度
            max_len_batch = max([len(v) for v in batch_text])
            if max_len_batch <= 15:
                max_len_batch = 15
            sentence = torch.tensor([pad([word2index.get(word, word2index.get(UNK)) for word in words],max_len_batch) for words in batch_text],device=device)
            #预测
            with torch.no_grad():
                model.eval()
                start = time.time()
                y_hat = model(sentence,seq_len)
                #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                #print(y_hat)  
                #threshold = opt.threshold
                threshold = 0.0
                with open('./DC/data/labels2index.pkl', 'rb') as f:
                    labels2index = pkl.load(f)
                other_index = labels2index['背景']
                y_hat = torch.softmax(y_hat, dim = 1)
                label = torch.argmax(y_hat,dim= 1 ).cpu().numpy().tolist()
                for i,item in enumerate((torch.max(y_hat,dim=1)[0] > threshold).cpu().numpy().tolist()):
                    if item == 0: 
                        label[i] = other_index 
                    else:
                        continue
                #label = y_hat.argmax(dim=1)
                #输出预测所用时间
                print("Prediction cost %3f seconds...." % (time.time()-start))
                print("这一批次预测了%d条数据" % len(label))
                #输出该批量文本的类别标签
                #label = label.cpu().numpy().tolist()
                label_list = [index2labels[i] for i in label]
                #print(label_list)
                with open('行政机关_result.txt', encoding='utf-8', mode='a') as f:
                    for i in zip(batch_text,label_list):
                        if i[1] != '行政机关':
                            print(i)
                            count += 1
                        else:
                            save_list.append(i[0])

                        f.write(i[0]+';'+i[1]+'\n')
        print("=========count===========",count)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%", len(save_list))                    
        #with open('建筑名称_new.txt', 'w', encoding='utf-8') as f:
        #    for i in save_list:
        #        f.write(i + '\n')
    #预测单条文本
    elif text != None:
        print('text is :', text)
        max_len_batch = len(text)
        #sentence = torch.tensor([word2index.get(word,word2index.get(UNK)) for word in text],device=device)
        if max_len_batch <= 15:
            max_len_batch = 15
        sentence = torch.tensor([pad([word2index.get(word, word2index.get(UNK)) for word in text],max_len_batch)],device=device)
        import pdb; pdb.set_trace()
        with torch.no_grad():
            model.eval()
            #print(sentence)
            x_test = sentence.view((1,-1))
            #x_test = sentence
            #print("x_test")
            #print(x_test)
            length = len(sentence)
            seq = torch.tensor([length])
            y_hat = model(x_test,seq)
            print('====',y_hat)
            y_hat = torch.softmax(y_hat, dim = 1)
            print(y_hat)
            label = torch.argmax(y_hat,dim=1)
            print(label)
            threshold = 0.0
            with open('./DC/data/labels2index.pkl', 'rb') as f:
                labels2index = pkl.load(f)
            other_index = labels2index['背景']
            
            if torch.max(y_hat) > threshold:
                label = label.item()
            else: 
                label = other_index                  

            #输出新文本的类别标签
            print("该文本被预测分类为:")
            print(index2labels[label])
 
    elif text == None and prediction_path == None:
        print("请输入单条预测文本或者预测文件路径哟~")
