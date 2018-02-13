import tensorflow as tf
import dev_supervision
from SAR_tf import gen_examples
from utils import load_vocab,gen_embeddings,load_data,convert2index
from sklearn.metrics import accuracy_score
from Encoder_tf import RNN_encoder
import AttentionLayer_tf
import logging
import argparse


class SAR():
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, dropout_rnn_in, dropout_rnn_out, bidirection=False, option_number=4):
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout_rnn_in = dropout_rnn_in
        self.dropout_rnn_out =dropout_rnn_out
        self.embedding_size = embedding_size
        self.embedding_layer = tf.get_variable("embedding", [vocab_size + 2, embedding_size], trainable=True,
                                               initializer=tf.constant_initializer(ini_weight))
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.p2q_attention = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_p2q')
        self.o2p_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_o2p')
        self.passage_encoder = RNN_encoder(hidden_dim, 'passage_encoder', bidirection, input_keep_prob=self.dropout_rnn_in,
                                           output_keep_prob = self.dropout_rnn_out,reuse=tf.AUTO_REUSE)
        self.question_encoder = RNN_encoder(hidden_dim, 'question_encoder', bidirection, input_keep_prob=self.dropout_rnn_in,
                                            output_keep_prob=self.dropout_rnn_out)
        self.option_encoder = RNN_encoder(hidden_dim, 'option_encoder', bidirection, input_keep_prob=self.dropout_rnn_in,
                                          output_keep_prob=self.dropout_rnn_out)
        self.gated_encoder1 = RNN_encoder(hidden_dim, 'g1_encoder', bidirection, input_keep_prob=self.dropout_rnn_in,
                                          output_keep_prob=self.dropout_rnn_out,reuse=tf.AUTO_REUSE)

        self.Wa = tf.get_variable('Wa',[self.weight_mtx_dim,4],initializer=tf.contrib.layers.xavier_initializer())

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt):
        v_passage = tf.nn.embedding_lookup(self.embedding_layer, passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer, qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer, opt)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p)
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])

        p2q_align = self.p2q_attention.score(encoded_passage, qst_ht)  # batch x sentence_len

        #p2q_align = tf.expand_dims(p2q_align,axis=2)
        p2q_align = p2q_align * tf.squeeze(msk_p)
        p2q_align = p2q_align / tf.expand_dims(tf.reduce_sum(p2q_align, axis=1),axis=1)

        p_expectation = tf.reduce_sum(tf.expand_dims(p2q_align,axis=2) * encoded_passage, axis=1)  # encoded_passage: batch x sentence_len x emb_dim
        # print p_expectation

        o2p_align = self.o2p_attention.score(opt_ht,p_expectation)
        #o2p_align = tf.matmul(p_expectation,self.Wa)
        return o2p_align

class Heuristic():
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, dropout_rnn_in, dropout_rnn_out, bidirection = False,option_number=4):
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_rnn_in = dropout_rnn_in
        self.dropout_rnn_out = dropout_rnn_out
        self.embedding_layer = tf.get_variable("embedding", [vocab_size + 2, embedding_size], trainable=True,
                                          initializer=tf.constant_initializer(ini_weight))
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.p2qa_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim,'W_p2qa')
        self.p2opt_attention = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim, 'W_q2opt')
        self.e2opt_attention = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_e2opt')
        self.e2opt_dot1 = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_e2opt_dot1')
        self.e2opt_dot2 = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_e2opt_dot2')

        self.passage_encoder = RNN_encoder(hidden_dim, 'passage_encoder', bidirection,
                                           input_keep_prob=self.dropout_rnn_in,
                                           output_keep_prob=self.dropout_rnn_out, reuse=tf.AUTO_REUSE)
        self.question_encoder = RNN_encoder(hidden_dim, 'question_encoder', bidirection,
                                            input_keep_prob=self.dropout_rnn_in,
                                            output_keep_prob=self.dropout_rnn_out)
        self.option_encoder = RNN_encoder(hidden_dim, 'option_encoder', bidirection,
                                          input_keep_prob=self.dropout_rnn_in,
                                          output_keep_prob=self.dropout_rnn_out)
        self.evidence_encoder = RNN_encoder(hidden_dim,'evidence_encoder',bidirection,input_keep_prob=self.dropout_rnn_in,
                                          output_keep_prob=self.dropout_rnn_out)

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt,
                evd, msk_evd, lst_evd, evd_score, tao):

        v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)
        v_evd = tf.nn.embedding_lookup(self.embedding_layer,evd)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        #evd_ht, encoded_evidence = self.evidence_encoder.encode(v_evd, lst_evd)
        #encoded_evidence = self.evd_selfatt.apply_self_attention(encoded_evidence)
        #e2opt_att = self.e2opt_attention.score(encoded_evidence,opt_ht)
        #evd_ht = tf.reduce_sum(tf.expand_dims(e2opt_att,axis=2) * encoded_evidence,axis=1)

        evd_score = evd_score / tao
        evd_score = tf.nn.softmax(tf.reshape(evd_score,[-1, self.option_number]))
        evd_score = tf.reshape(evd_score,[-1, self.option_number, 1])

        #evd_score = tf.reshape(evd_score, [-1, self.option_number, 1])
        #evd_score = evd_score / tf.expand_dims(tf.reduce_sum(evd_score, axis=1), axis=1)

        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        #evd_ht = tf.reshape(evd_ht, [-1, self.option_number, self.weight_mtx_dim])
        #evd_ht = evd_ht * evd_score
        qst_ht = tf.expand_dims(qst_ht, 1)

        qstopt_ht = opt_ht + qst_ht

        #probs_qstopt = self.e2opt_dot1.score(evd_ht,qstopt_ht)
        #probs_opt = self.e2opt_dot2.score(evd_ht,opt_ht)

        msk_p = tf.expand_dims(msk_p, 1)
        p2qa_align = self.p2qa_attention.score(encoded_passage,opt_ht)  # p2qa_align: batch x question_num x sentence_len
        p2qa_align = p2qa_align * msk_p
        p2qa_align = p2qa_align / tf.expand_dims(tf.reduce_sum(p2qa_align, axis=2), axis=2)
        pqa_expectation = tf.matmul(p2qa_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim

        expectation = pqa_expectation * evd_score
        o2q_align = self.p2opt_attention.score(expectation, qstopt_ht)

        return o2q_align

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-test", type=str, help="path to test data")
    arg_parser.add_argument("-sar", type=str, help='path to sar', default='')
    arg_parser.add_argument("-gar", type=str, help='path to gar', default='')
    arg_parser.add_argument("-heuristic", type=str, help='path to heuristic', default='')

    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info('-' * 20 + 'loading training data and vocabulary' + '-' * 20)
    batch_size = 32

    vocab = load_vocab('RACE/dict.pkl')
    # vocab = utils.build_dict(train_data[0] + train_data[1] + train_data[2])
    vocab_len = len(vocab.keys())
    embedding_size = 100
    hidden_size = 128
    init_embedding = gen_embeddings(vocab, embedding_size, 'RACE/glove.6B/glove.6B.100d.txt')

    print vocab_len

    if args.sar:
        test_data = load_data(args.test)
        test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab, sort_by_len=False)
        all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y, 32)
        g_sar = tf.Graph()
        with g_sar.as_default():
            article = tf.placeholder(tf.int32,(None,None))
            msk_a = tf.placeholder(tf.float32,(None,None))
            lst_a = tf.placeholder(tf.int32,(None))

            qst = tf.placeholder(tf.int32,(None,None))
            msk_qst = tf.placeholder(tf.float32,(None,None))
            lst_qst = tf.placeholder(tf.int32,(None,))

            opt = tf.placeholder(tf.int32,(None,None))
            msk_opt = tf.placeholder(tf.float32,(None,None))
            lst_opt = tf.placeholder(tf.int32,(None))

            dropout_rnn_in = tf.placeholder(tf.float32,())
            dropout_rnn_out = tf.placeholder(tf.float32, ())

            y = tf.placeholder(tf.int32, (None))

            sar = SAR(vocab_size=vocab_len, embedding_size=embedding_size, ini_weight=init_embedding,
                      dropout_rnn_in=dropout_rnn_in, dropout_rnn_out=dropout_rnn_out, hidden_dim=hidden_size, bidirection=True)

            sar_probs = sar.forward(article, msk_a, lst_a, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt)

            sar_probs = tf.nn.softmax(sar_probs)
            sar_one_best = tf.argmax(sar_probs, axis=1)
            sar_session = tf.Session()
            print tf.trainable_variables()
            saver = tf.train.Saver()

            saver.restore(sar_session, args.sar)
            predicts = []
            gold = []

            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_y) in enumerate(all_test):
                # Evaluate on the dev set
                train_correct = 0
                [pred_this_instance, probs_this_batch] = sar_session.run([sar_one_best, sar_probs],
                                                 feed_dict={article: mb_x1, msk_a: mb_mask1,lst_a: mb_lst1,
                                                            qst: mb_x2, msk_qst: mb_mask2,lst_qst: mb_lst2,
                                                            opt: mb_x3, msk_opt: mb_mask3,lst_opt: mb_lst3,
                                                            y: mb_y, dropout_rnn_in: 1.0,dropout_rnn_out: 1.0})
                predicts += list(pred_this_instance)
                gold += mb_y

            dev_acc = accuracy_score(gold, predicts)
            print dev_acc

    if args.heuristic:
        test_data = dev_supervision.load_data(args.test)
        test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_y = \
            dev_supervision.convert2index(test_data, vocab, sort_by_len=False)
        all_test = dev_supervision.gen_examples(test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_y, 32)
        g_heuristic = tf.Graph()
        with g_heuristic.as_default():
            article = tf.placeholder(tf.int32, (None, None))
            msk_a = tf.placeholder(tf.float32, (None, None))
            lst_a = tf.placeholder(tf.int32, (None))

            qst = tf.placeholder(tf.int32, (None, None))
            msk_qst = tf.placeholder(tf.float32, (None, None))
            lst_qst = tf.placeholder(tf.int32, (None,))

            opt = tf.placeholder(tf.int32, (None, None))
            msk_opt = tf.placeholder(tf.float32, (None, None))
            lst_opt = tf.placeholder(tf.int32, (None))

            dropout_rnn_in = tf.placeholder(tf.float32, ())
            dropout_rnn_out = tf.placeholder(tf.float32, ())

            y = tf.placeholder(tf.int32, (None))

            evd = tf.placeholder(tf.int32, (None, None))
            msk_evd = tf.placeholder(tf.float32, (None, None))
            lst_evd = tf.placeholder(tf.int32, (None))

            evd_score = tf.placeholder(tf.float32, (None))
            tao = tf.placeholder(tf.float32, (None))

            heuristic = Heuristic(vocab_size=vocab_len, embedding_size=embedding_size, ini_weight=init_embedding,
                                  dropout_rnn_in=dropout_rnn_in, dropout_rnn_out=dropout_rnn_out, hidden_dim=hidden_size,
                                  bidirection=True)

            heu_probs = tf.nn.softmax(heuristic.forward(article, msk_a, lst_a, qst, msk_qst, lst_qst, opt, msk_opt,
                                                        lst_opt, evd, msk_evd, lst_evd, evd_score, tao))
            heu_one_best = tf.argmax(heu_probs, axis=1)

            heu_sessioin = tf.Session()
            saver = tf.train.Saver()
            saver.restore(heu_sessioin, args.heuristic)
            predicts = []
            gold = []

            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_x4, mb_mask4, mb_lst4, mb_s, mb_y) in enumerate(all_test):
                # Evaluate on the dev set
                train_correct = 0
                [pred_this_instance, probs_this_batch] = \
                    heu_sessioin.run([heu_one_best, heu_probs], feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                          qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                          opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                          evd: mb_x4, msk_evd: mb_mask4, lst_evd: mb_lst4,
                                                          evd_score: mb_s, y: mb_y, dropout_rnn_in: 1.0,
                                                          dropout_rnn_out: 1.0, tao: 1.0})
                predicts += list(pred_this_instance)
                gold += mb_y

            dev_acc = accuracy_score(gold, predicts)
            print dev_acc
