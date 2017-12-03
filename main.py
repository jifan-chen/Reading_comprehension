import torch
from torch import nn,optim
from torch.autograd import Variable
import attention_layer
import test_encoder
from itertools import ifilter
import utils
from sklearn.metrics import f1_score,accuracy_score

from utils import *
from preprocessing import *


class Trainer(nn.Module):

    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, option_number=4):
        super(Trainer, self).__init__()
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size+1,embedding_size)
        self.embedding_layer.weight = nn.Parameter(ini_weight)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.passage_encoder = test_encoder.RNNEncoder(embedding_size,'GRU',True,1,True,hidden_dim)
        self.query_encoder = test_encoder.RNNEncoder(embedding_size,'GRU',True,1,True,hidden_dim)
        self.option_encoder = test_encoder.RNNEncoder(embedding_size,'GRU',True,1,True,hidden_dim)
        self.passage_encoder2 = torch.nn.GRU(hidden_dim*2,hidden_dim,batch_first=True,bidirectional=True)
        self.p2q_attention = attention_layer.BilinearAttentionP2Q(hidden_dim * 2 )
        self.o2p_attention = attention_layer.BilinearAttentionO2P(hidden_dim * 2)

    def forward(self,passage,msk_p,lst_p,question,msk_q,lst_q,option,msk_o,lst_o):
        v_passage = self.embedding_layer(passage)
        #v_passage = self.emb_dropout(v_passage)
        v_question = self.embedding_layer(question)
        #v_question = self.emb_dropout(v_question)
        v_option = self.embedding_layer(option)
        #v_option = self.emb_dropout(v_option)

        p_ht,encoded_passage = self.passage_encoder(v_passage,msk_p,lst_p)
        q_ht,encoded_question = self.query_encoder(v_question,msk_q,lst_q)
        o_ht,encoded_option = self.option_encoder(v_option,msk_o,lst_o)

        o_ht = o_ht.view(-1,self.option_number,self.hidden_dim *2 )
        p_ht = torch.unsqueeze(p_ht, 0)
        p_ht,encoded_passage = self.passage_encoder2(p_ht)
        p_ht = torch.squeeze(p_ht,0)
        
        p2q_align = self.p2q_attention(p_ht,q_ht)

        #print p_ht
        #print p2q_align
        #print p2q_align.data + 1
        #print p_ht
        #p_expectation = (p2q_align.transpose(0,1) * p_ht).sum(0)
        #p_original = p_ht.mean(dim=0).unsqueeze(0).repeat(q_ht.data.shape[0],1)
        #print p_expectation
        p_expectation = torch.mm(p2q_align,p_ht)
        #print p_expectation
        #p_expectation = torch.unsqueeze(p_expectation,0)

        #o2p_align = self.o2p_attention(o_ht,p_expectation)
        o2p_align = self.o2p_attention(o_ht,p_expectation)
        #print o2p_align
        return o2p_align


if __name__ == '__main__':

    print '******* loading training data and vocabulary *********'
    training_data = load_data('RACE/train/processed_data_middle.pkl')
    test_data = load_data('RACE/test/processed_data_middle.pkl')
    vocab = load_vocab('RACE/train/vocab_middle.pkl')
    vocab_len = len(vocab.keys())

    print vocab_len
    embedding_size = 100
    hidden_size = 128
    pre_embedding = load_pretrained_embedding('RACE/glove.6B/glove.6B.100d.txt')
    init_embedding = init_embedding_matrix(vocab,pre_embedding,embedding_size)
    trainer = Trainer(vocab_size=vocab_len,embedding_size=embedding_size,ini_weight=init_embedding,hidden_dim=hidden_size)
    params = trainer.parameters()
    #params = ifilter(lambda p: p.requires_grad, trainer.parameters())
    #for param in params:
    #    print param
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params,lr=0.1)
    train_loss = 0
    num_epoch = 50
    sample_count = 0
    print '***********training***********'
    for epoch in range(num_epoch):
        predicts = []
        gold = []
        trainer.train()
        for it,p in enumerate(training_data):
            article,mask_a,lastpst_a = p['article']
            article = Variable(torch.LongTensor(article))
            mask_a = Variable(torch.FloatTensor(mask_a))
            lastpst_a = Variable(torch.LongTensor(lastpst_a))

            q,msk_q,lst_q = p['questions']
            o,msk_o,lst_o = p['options']
            targets = p['answers']

            q = Variable(torch.LongTensor(q))
            msk_q = Variable(torch.FloatTensor(msk_q))
            lst_q = Variable(torch.LongTensor(lst_q))

            o = Variable(torch.LongTensor(o))
            msk_o = Variable(torch.FloatTensor(msk_o))
            lst_o = Variable(torch.LongTensor(lst_o))

            y = Variable(torch.LongTensor(targets))

            optimizer.zero_grad()
            probs = trainer(article,mask_a,lastpst_a,q,msk_q,lst_q,o,msk_o,lst_o)

            loss = loss_function(probs,y)
            loss.backward()
            train_loss += loss.data[0]

            sample_count += 1
            optimizer.step()

            prob, predict = torch.max(probs, 1)
            predicts.append(predict.data.numpy()[0])
            gold.append(y.data.numpy()[0])

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
        for p in test_data:
            article, mask_a, lastpst_a = p['article']
            article = Variable(torch.LongTensor(article))
            mask_a = Variable(torch.FloatTensor(mask_a))
            lastpst_a = Variable(torch.LongTensor(lastpst_a))

            q, msk_q, lst_q = p['questions']
            o, msk_o, lst_o = p['options']
            targets = p['answers']

            q = Variable(torch.LongTensor(q))
            msk_q = Variable(torch.FloatTensor(msk_q))
            lst_q = Variable(torch.LongTensor(lst_q))

            o = Variable(torch.LongTensor(o))
            msk_o = Variable(torch.FloatTensor(msk_o))
            lst_o = Variable(torch.LongTensor(lst_o))

            y = Variable(torch.LongTensor(targets))
            probs = trainer(article, mask_a, lastpst_a, q, msk_q, lst_q, o, msk_o, lst_o)

            loss = loss_function(probs, y)
            test_loss += loss.data[0]

            prob, predict = torch.max(probs, 1)
            predicts.append(predict.data.numpy()[0])
            gold.append(y.data.numpy()[0])

        print '*****Test Error*****' + str(test_loss / len(test_data))
        print accuracy_score(gold,predicts)
        torch.save(trainer.state_dict(),'model'+str(epoch))



