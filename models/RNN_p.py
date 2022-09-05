from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule
import numpy as np


class Model(BasicModule):#继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module
    def __init__(self, vocab_size,config):#opt是config类的实例 里面包括所有模型超参数的配置
        super(Model, self).__init__()
        self.model_name = 'RNN_p'
        #嵌入层
        if config.pretrained_embedding_matrix_path is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            embedding_pretrained = torch.tensor(np.load(config.pretrained_embedding_matrix_path)["embeddings"].astype('float32'))
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=config.frozen)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(vocab_size, config.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调

        # bidirectional设为True即得到双向循环神经网络
        self.rnn = nn.LSTM(input_size=config.embed_size,
        # self.encoder = nn.GRU(input_size=opt.embed_size,
                               hidden_size=config.recurrent_hidden_size,
                               num_layers=config.num_layers,
                               bidirectional=config.bidirectional,
                               batch_first=True,
                               dropout=config.drop_prop
                               )
        self.fc = nn.Linear(2 * config.recurrent_hidden_size, config.classes) #最终时间步的隐藏状态作为全连接层输入
      #变长序列处理 
    def forward(self,x,seq):
        s = seq
        x = self.embedding(x)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, s, batch_first = True,enforce_sorted=False)
        out, (hn,_) = self.rnn(packed) #(batch_size,seq_len,hidden_size)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  
        #out = out[:,-1,:]
        out = torch.cat((hn[2], hn[3]), -1)
        out = self.fc(out)
        
        return out
