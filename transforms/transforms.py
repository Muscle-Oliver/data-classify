from utils import get_and_format_data, get_index, build_vocab
import pickle as pkl
import os
import random
import copy
import numpy as np


MAX_VOCAB_SIZE = 20000  
UNK, PAD = '<UNK>', '<PAD>'

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BasicTransform(object):
    def __init__(self, vocab_path, train_path):
        self.vocab_path = vocab_path
        self.train_path = train_path
    def __call__(self, data):
        # print('data',data)
        # print('Basic transform',data)
        tokenizer = lambda x: [y for y in x]        
        #构建词/字典
        if os.path.exists(self.vocab_path): #如果存在构建好的词/字典 则加载
        # if vocab_path:
            # print('加载已存在的字典')
            vocab = pkl.load(open(self.vocab_path, 'rb'))
        else:  #构建词/字典（基于训练集）
            # print('没有存在的字典，现在生成并保存')
            vocab = build_vocab(self.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            #保存构建好的词/字典
            pkl.dump(vocab, open(config.vocab_path, 'wb'))
        #convert text to index_list
        text_index = get_index(data, tokenizer, vocab)
        return text_index   

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NameTransform(object):
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab
    def __call__(self, data):
        #unusual_familyname = ['易','祝','姬','欧阳']
        unusual_familyname = ['顾', '侯', '邵', '孟', '龙', '万', '段', '漕', '钱', '汤', '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文', '庞', '樊', '兰', '殷', '施', '陶', '洪', '翟', '安', '颜', '倪', '严', '牛', '温', '芦', '季', '俞', '章', '鲁', '葛', '伍', '韦', '申', '尤', '毕', '聂', '丛', '焦', '向', '柳', '邢', '路', '岳', '齐', '沿', '梅', '莫', '庄', '辛', '管', '祝', '左', '涂', '谷', '祁', '时', '舒', '耿', '牟', '卜', '路', '詹', '关', '苗', '凌', '费', '纪', '靳', '盛', '童', '欧', '甄', '项', '曲', '成', '游', '阳', '裴', '席', '卫', '查', '屈', '鲍', '位', '覃', '霍', '翁', '隋', '植', '甘', '景', '薄', '单', '包', '司', '柏', '宁', '柯', '阮', '桂', '闵', '欧阳', '解', '强', '柴', '华', '车', '冉', '房', '边', '辜', '吉', '饶', '刁', '瞿', '戚', '丘', '古', '米', '池', '滕', '晋', '苑', '邬', '臧', '畅', '宫', '来', '嵺', '苟', '全', '褚', '廉', '简', '娄', '盖', '符', '奚', '木', '穆', '党', '燕', '郎', '邸', '冀', '谈', '姬', '屠', '连', '郜', '晏', '栾', '郁', '商', '蒙', '计', '喻', '揭', '窦', '迟', '宇', '敖', '糜', '鄢', '冷', '卓', '花', '仇', '艾', '蓝', '都', '巩', '稽', '井', '练', '仲', '乐', '虞', '卞', '封', '竺', '冼', '原', '官', '衣', '楚', '佟', '栗', '匡', '宗', '应', '台', '巫', '鞠', '僧', '桑', '荆', '谌', '银', '扬', '明', '沙', '薄', '伏', '岑', '习', '胥', '保', '和', '蔺']

        if data[1] == 2 and random.random() > 0.5:
            #print("11111111111", data)
            random_familyname = random.choice(unusual_familyname)
            #print("2222222222", random_familyname)
            random_familyname_index = self.vocab.get(random_familyname, self.vocab.get(UNK))
            #print("3333333333", random_familyname_index)
            data[0][0] =  random_familyname_index
            #print("44444444444", data)
        return data   

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TargetTransform(object):
    def __init__(self, labels2index_path):
        self.labels2index_path = labels2index_path
    def __call__(self, target):
        # print('target transform',target)
        all_targets = []
        label_to_index = pkl.load(open(self.labels2index_path, 'rb'))
        label_index = label_to_index[target]
        return label_index

    def __repr__(self):
        return self.__class__.__name__ + '()'

