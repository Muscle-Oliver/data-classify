import pickle as pkl
import json

with open('./DC/data/labels2index.pkl', 'rb') as f:
    labels2index = pkl.load(f)

print(labels2index)
labels = list(labels2index.keys())
print(labels)

with open('./DC/data/labels2index.json','w') as f:
    json.dump(labels2index,f)

with open('./DC/data/labels.json','w') as f:
    json.dump(labels,f)

with open('./DC/data/index2labels.pkl', 'rb') as f:
    index2labels = pkl.load(f)
print(index2labels)

with open('./DC/data/index2labels.json','w') as f:
    json.dump(index2labels,f) 

with open('./DC/data/vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

print(vocab)

vocab.update({'UNK':vocab.pop("<UNK>")})
vocab.update({'PAD':vocab.pop("<PAD>")})
print('======')
print(vocab)
vocab_size = len(vocab)

with open('./DC/data/word2index.json','w') as f:
    json.dump(vocab,f)
with open('./DC/data/vocabsize.json','w') as f:
    json.dump(vocab_size,f)