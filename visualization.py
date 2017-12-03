from utils import *
from preprocessing import *
#from main import Trainer
import attention_layer
import test_encoder
from torch.autograd import Variable
from sklearn.metrics import f1_score,accuracy_score
from torch import nn
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
import attention_layer
import test_encoder
import logging
from itertools import ifilter
import utils
from sklearn.metrics import accuracy_score

from utils import *
from preprocessing import *


class Trainer(nn.Module):
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, option_number=4):
        super(Trainer, self).__init__()
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size + 1, embedding_size)
        self.embedding_layer.weight = nn.Parameter(ini_weight)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.passage_encoder = test_encoder.RNNEncoder(embedding_size, 'GRU', False, 1, True, hidden_dim)
        self.qstopt_encoder = test_encoder.RNNEncoder(embedding_size, 'GRU', False, 1, True, hidden_dim)
        self.p2qa_attention = attention_layer.BilinearAttentionP2QA(hidden_dim )
        self.option_dot_product = attention_layer.DotProductAttention(hidden_dim )

    def forward(self, passage, msk_p, lst_p, qstopt,msk_qstopt,lst_qstopt):
        v_passage = self.embedding_layer(passage)
        v_qstopt = self.embedding_layer(qstopt)

        p_ht, encoded_passage = self.passage_encoder(v_passage, msk_p, lst_p)
        qstopt_ht, encoded_question = self.qstopt_encoder(v_qstopt, msk_qstopt, lst_qstopt)
        qstopt_ht = qstopt_ht.view(-1, self.option_number, self.hidden_dim )

        p2qa_align = self.p2qa_attention(encoded_passage,qstopt_ht)

        p_expectation = torch.bmm(p2qa_align, encoded_passage)
        #print p_expectation

        o2p_align = self.option_dot_product(p_expectation)
        return o2p_align,p2qa_align

def gen_examples(x1, x2, x3, x4, y, question_belong,batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x4 = [x4[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_qst = [question_belong[t] for t in minibatch]
        mb_x1, mb_mask1, mb_lst1 = pad_sequences(mb_x1)
        #mb_x2, mb_mask2, mb_lst2 = utils.pad_sequences(mb_x2)
        #mb_x3, mb_mask3, mb_lst3 = utils.pad_sequences(mb_x3)
        mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x4, mb_mask4, mb_lst4, mb_y, mb_qst))
    return all_ex

def load_data(in_file, max_example=None, relabeling=True):

    documents = []
    questions = []
    answers = []
    options = []
    qs_op = []
    question_belong = []
    num_examples = 0

    def get_file(path):
        files = []
        for inf in os.listdir(path):
            new_path = os.path.join(path, inf)
            if os.path.isdir(new_path):
                assert inf in ["middle", "high"]
                files += get_file(new_path)
            else:
                if new_path.find(".DS_Store") != -1:
                    continue
                files += [new_path]
        return files
    files = get_file(in_file)

    for inf in files:
        obj = json.load(open(inf, "r"))

        for i, q in enumerate(obj["questions"]):
            question_belong += [inf]
            documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            for j in range(4):
                #print obj['options'][i][j]
                qs_op += [q + " " + obj['options'][i][j]]
            options += obj["options"][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break

    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list

    documents = clean(documents)
    questions = clean(questions)
    options = clean(options)
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options,qs_op,answers,question_belong)

if __name__ == '__main__':

    print '-'*20,'loading training data and vocabulary','-'*20
    vocab = load_vocab('RACE/dict.pkl')
    vocab_len = len(vocab.keys())
    print vocab_len
    batch_size = 32

    # data loaded order: doc, question, option, Qst+Opt, Answer
    train_data = load_data('RACE/data/train/middle/')

    train_x1, train_x2, train_x3, train_x4, train_y = convert2index(train_data[:-1], vocab,sort_by_len=False)
    question_belong = train_data[-1]
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_y, question_belong, 32)
    mb_qst = all_train[-1]
    print '-'*20,'Done','-'*20

    embedding_size = 100
    hidden_size = 128
    pre_embedding = load_pretrained_embedding('RACE/glove.6B/glove.6B.100d.txt')
    init_embedding = init_embedding_matrix(vocab,pre_embedding,embedding_size)
    trainer = Trainer(vocab_size=vocab_len,embedding_size=embedding_size,ini_weight=init_embedding,hidden_dim=hidden_size)
    trainer.load_state_dict(torch.load('model35'))

    train_loss = 0
    num_epoch = 50
    sample_count = 0
    loss_function = nn.CrossEntropyLoss()
    trainer.eval()
    predicts = []
    gold = []
    question_idx = []
    attentions = []

    for it,(mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_y, qst) in enumerate(all_train):
        article = Variable(torch.LongTensor(mb_x1))
        mask_a = Variable(torch.FloatTensor(mb_mask1))
        lst_a = Variable(torch.LongTensor(mb_lst1))

        qsopt = Variable(torch.LongTensor(mb_x2))
        msk_qsopt = Variable(torch.FloatTensor(mb_mask2))
        lst_qsopt = Variable(torch.LongTensor(mb_lst2))

        y = Variable(torch.LongTensor(mb_y))

        probs,attention_weight = trainer(article,mask_a,lst_a,qsopt,msk_qsopt,lst_qsopt)
        attentions.append(attention_weight.data.numpy())
        question_idx.append(range(it*batch_size,(it+1)*batch_size))
        #print question_idx
        #print probs
        loss = loss_function(probs,y)
        #print loss
        train_loss += loss.data[0]
        #print attention_weight
        sample_count += 1
        #print mb_question
        prob, predict = torch.max(probs, 1)
        predicts.append(predict.data.numpy()[0])
        gold.append(y.data.numpy()[0])

    print '*****Test Error*****' + str(train_loss / sample_count)
    print accuracy_score(gold,predicts)
