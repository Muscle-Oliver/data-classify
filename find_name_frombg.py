import jieba
from config import opt
from importlib import import_module
import pickle as pkl
import torch

UNK, PAD = '<UNK>', '<PAD>'

name_data = []

#加载之前预处理过程 保存的词到索引的映射字典
with open('vocab.pkl', 'rb') as f:
    word2index = pkl.load(f)

# 加载之前预处理过程 保存的索引到类别标签的映射字典
with open('index2labels.pkl', 'rb') as f:
    index2labels = pkl.load(f)

model_name = 'RNN_p'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = import_module('models.' + model_name)
config = opt
model = x.Model(len(index2labels), config).to(config.device) #构建模型对象 并to_device

model.load_state_dict(torch.load(model_name + '.pth',map_location=device))

with open('./source/background_data_2.txt',encoding = 'utf-8', mode='r') as f:
    lines = f.read().strip().split('\n')
    for line in lines:
        # print(line)
        l = jieba.lcut(line)
        for i in l:
            sentence = torch.tensor([word2index.get(word,word2index.get(UNK)) for word in i],device=device)
            with torch.no_grad():
                model.eval()
                x_test = sentence.view((1,-1))
                length = len(sentence)
                seq = torch.tensor([length])
                y_hat = model(x_test,seq)
                label = torch.argmax(y_hat,dim=1)            
                if  index2labels[label.item()] == '姓名' and len(i) > 1 :
                    print('@@@@@', i)
                    name_data.append(i)

name_data = list(set(name_data))
print('name_data', len(name_data))

with open('./source/name_inbg.txt',encoding = 'utf-8', mode='w') as f:
    for line in name_data:
        f.write(line+'\n')


with open('./source/background_name.txt',encoding = 'utf-8', mode='w') as f:
    for line in name_data:
        f.write(line[0]+'\n')