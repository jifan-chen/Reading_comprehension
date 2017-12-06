import numpy as np
import operator

import torch

def load_pretrained_embedding(emb_url):
    f = open(emb_url)
    pretrained_emb = {}
    for line in f:
        data = line.split()
        word = data[0]
        vector = data[1:]
        pretrained_emb[word] = vector

    return pretrained_emb

def init_embedding_matrix(vocab,pretrained_embedding,emb_size):
    init_embedding_matrix = np.random.randn(len(vocab.keys())+1,emb_size)
    #init_embedding_matrix = np.random.uniform(-0.01,0.01,[len(vocab.keys())+1,emb_size])
    sorted_vocab = sorted(vocab.items(),key=operator.itemgetter(1))
    count = 0
    print sorted_vocab[:10]
    for i in range(1,len(vocab.keys())+1):
        try:
            init_embedding_matrix[i] = np.array(pretrained_embedding[sorted_vocab[i-1][0]])
        except KeyError:
            count += 1
            #print sorted_vocab[i-1][0]
            pass
    print count
    return init_embedding_matrix

if __name__ == '__main__':
    emb = load_pretrained_embedding('RACE/glove.6B/glove.6B.50d.txt')
    print emb['the']