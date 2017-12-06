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
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, bidirection = False, option_number=4):
        super(Trainer, self).__init__()
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size + 1, embedding_size)
        self.embedding_layer.weight = nn.Parameter(ini_weight)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.passage_encoder = test_encoder.RNNEncoder(embedding_size, 'GRU', bidirection, 1, True, hidden_dim)
        self.qst_encoder = test_encoder.RNNEncoder(embedding_size, 'GRU', bidirection, 1, True, hidden_dim)
        self.opt_encoder = test_encoder.RNNEncoder(embedding_size, 'GRU', bidirection, 1, True, hidden_dim)
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.p2qa_attention = attention_layer.BilinearAttentionP2QA(self.weight_mtx_dim )
        self.option_dot_product = attention_layer.DotProductAttention(self.weight_mtx_dim )

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt):
        v_passage = self.embedding_layer(passage)
        v_qst = self.embedding_layer(qst)
        v_opt = self.embedding_layer(opt)

        p_ht, encoded_passage = self.passage_encoder(v_passage, msk_p, lst_p)
        qst_ht, encoded_question = self.qst_encoder(v_qst, msk_qst, lst_qst)
        opt_ht, encoded_option = self.opt_encoder(v_opt, msk_opt, lst_opt)

        opt_ht = opt_ht.view(-1, self.option_number, self.weight_mtx_dim)
        qst_ht = torch.unsqueeze(qst_ht,1)

        qstopt_ht = opt_ht + qst_ht
        #qstopt_ht = opt_ht

        #qst_ht = qst_ht.repeat(1,4,1)
        #qstopt_ht = torch.cat((opt_ht,qst_ht),dim=2)

        p2qa_align = self.p2qa_attention(encoded_passage,qstopt_ht)

        msk_p = torch.squeeze(msk_p)
        msk_p = torch.unsqueeze(msk_p,1)
        #p2qa_align = p2qa_align * msk_p

        #print p2qa_align.sum(dim=2)
        #print encoded_passage
        p_expectation = torch.bmm(p2qa_align, encoded_passage)
        #print p_expectation

        o2p_align = self.option_dot_product(p_expectation)

        return o2p_align

def gen_examples(x1, x2, x3, x4, y ,batch_size, concat=False):
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
        mb_x1, mb_mask1, mb_lst1 = pad_sequences(mb_x1)
        mb_x2, mb_mask2, mb_lst2 = utils.pad_sequences(mb_x2)
        mb_x3, mb_mask3, mb_lst3 = utils.pad_sequences(mb_x3)
        #mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3 ,mb_y))
    return all_ex

if __name__ == '__main__':

    print '-'*20,'loading training data and vocabulary','-'*20
    vocab = load_vocab('RACE/dict.pkl')
    vocab_len = len(vocab.keys())
    print vocab_len
    batch_size = 32

    # data loaded order: doc, question, option, Qst+Opt, Answer
    train_data= load_data('RACE/data/train/middle/')
    dev_data = load_data('RACE/data/dev/middle')
    test_data = load_data('RACE/data/test/middle/')

    train_x1, train_x2, train_x3, train_x4, train_y = convert2index(train_data, vocab,sort_by_len=False)
    dve_x1, dev_x2, dev_x3, dev_x4, dev_y = convert2index(dev_data, vocab,sort_by_len=False)
    test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab,sort_by_len=False)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_y, 32)
    all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y,32)

    print '-'*20,'Done','-'*20

    embedding_size = 100
    hidden_size = 128
    pre_embedding = load_pretrained_embedding('RACE/glove.6B/glove.6B.100d.txt')
    init_embedding = init_embedding_matrix(vocab,pre_embedding,embedding_size)
    init_embedding = torch.FloatTensor(init_embedding)
    trainer = Trainer(vocab_size=vocab_len,embedding_size=embedding_size,ini_weight=init_embedding,hidden_dim=hidden_size,bidirection=True)
    params = trainer.parameters()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params,lr=0.1)
    num_epoch = 50

    print '***********training***********'
    for epoch in range(num_epoch):
        predicts = []
        gold = []
        trainer.train()
        train_loss = 0
        sample_count = 0
        for it,(mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3,mb_y) in enumerate(all_test):
            article = Variable(torch.LongTensor(mb_x1))
            mask_a = Variable(torch.FloatTensor(mb_mask1))
            lst_a = Variable(torch.LongTensor(mb_lst1))

            qst = Variable(torch.LongTensor(mb_x2))
            msk_qst = Variable(torch.FloatTensor(mb_mask2))
            lst_qst = Variable(torch.LongTensor(mb_lst2))

            opt = Variable(torch.LongTensor(mb_x3))
            msk_opt = Variable(torch.FloatTensor(mb_mask3))
            lst_opt = Variable(torch.LongTensor(mb_lst3))

            y = Variable(torch.LongTensor(mb_y))

            optimizer.zero_grad()
            probs = trainer(article,mask_a,lst_a,qst,msk_qst,lst_qst,opt,msk_opt,lst_opt)
            #print probs
            loss = loss_function(probs,y)
            loss.backward()
            train_loss += loss.data[0]

            sample_count += 1
            optimizer.step()

            prob, predict = torch.max(probs, 1)
            predicts += list(predict.data.numpy())
            gold += list(y.data.numpy())

            if it % 100 == 0:
                print '*****Epoch:'+ str(epoch) + '*****iteration:' + str(it) + '*****loss:' +str(train_loss/sample_count)
                train_loss = 0
                sample_count = 0
        print accuracy_score(gold, predicts)
        # *********** test *************
        print "*****************testing***********************"

        trainer.eval()
        test_loss = 0
        predicts = []
        gold = []
        sample_count = 0
        for it,(mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3,mb_y) in enumerate(all_test):
            article = Variable(torch.LongTensor(mb_x1))
            mask_a = Variable(torch.FloatTensor(mb_mask1))
            lst_a = Variable(torch.LongTensor(mb_lst1))

            qst = Variable(torch.LongTensor(mb_x2))
            msk_qst = Variable(torch.FloatTensor(mb_mask2))
            lst_qst = Variable(torch.LongTensor(mb_lst2))

            opt = Variable(torch.LongTensor(mb_x3))
            msk_opt = Variable(torch.FloatTensor(mb_mask3))
            lst_opt = Variable(torch.LongTensor(mb_lst3))

            y = Variable(torch.LongTensor(mb_y))

            probs = trainer(article, mask_a, lst_a, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt)

            loss = loss_function(probs,y)
            test_loss += loss.data[0]

            sample_count += 1

            prob, predict = torch.max(probs, 1)
            predicts += list(predict.data.numpy())
            gold += list(y.data.numpy())

        print '*****Test Error*****' + str(test_loss / sample_count)
        print accuracy_score(gold,predicts)
        torch.save(trainer.state_dict(),'model'+str(epoch))
