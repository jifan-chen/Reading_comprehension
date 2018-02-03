import tensorflow as tf
from utils import load_vocab,gen_embeddings,load_data,convert2index
from develop_tf import gen_examples
from sklearn.metrics import accuracy_score
from Encoder_tf import RNN_encoder
import AttentionLayer_tf

class Trainer():
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, bidirection = False, option_number=4):
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = tf.get_variable("embedding", [vocab_size + 2, embedding_size], trainable=True,
                                          initializer=tf.constant_initializer(ini_weight))
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.p2qa_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim,'W_p2qa')
        self.option_dot_product = AttentionLayer_tf.DotProductAttention(self.weight_mtx_dim,'W_dotprod')
        self.p2opt_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim,'W_p2opt')
        self.q2opt_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_q2opt')

        self.passage_encoder = RNN_encoder(hidden_dim,'passage_encoder',bidirection,keep_prob=0.8)
        self.question_encoder = RNN_encoder(hidden_dim,'question_encoder',bidirection,keep_prob=0.8)
        self.option_encoder = RNN_encoder(hidden_dim,'option_encoder',bidirection,keep_prob=0.8)

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt,dropout_rate):
        v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        #v_passage = tf.nn.dropout(v_passage,dropout_rate)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        #v_qst = tf.nn.dropout(v_qst,dropout_rate)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)
        #v_opt = tf.nn.dropout(v_opt,dropout_rate)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        msk_p = tf.expand_dims(msk_p, 1)

        p2opt_align = self.p2opt_attention.score(encoded_passage, opt_ht)
        p2opt_align = p2opt_align * msk_p
        popt_expectation = tf.matmul(p2opt_align, encoded_passage)

        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht
        p2qa_align = self.p2qa_attention.score(encoded_passage,qstopt_ht)  # p2qa_align: batch x question_num x sentence_len
        p2qa_align = p2qa_align * msk_p
        pqa_expectation = tf.matmul(p2qa_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim

        #expectation = pqa_expectation + popt_expectation
        #expectation = pqa_expectation
        expectation = popt_expectation
        #o2p_align = self.option_dot_product.score(pqa_expectation)
        o2q_align = self.q2opt_attention.score(expectation,tf.squeeze(qst_ht))

        return o2q_align,p2opt_align


vocab = load_vocab('RACE/dict.pkl')
vocab_len = len(vocab.keys())
embedding_size = 100
hidden_size = 128
init_embedding = gen_embeddings(vocab, embedding_size, 'RACE/glove.6B/glove.6B.100d.txt')
trainer = Trainer(vocab_size=vocab_len,embedding_size=embedding_size,ini_weight=init_embedding,hidden_dim=hidden_size,bidirection=True)

article = tf.placeholder(tf.int32,(None,None))
msk_a = tf.placeholder(tf.float32,(None,None))
lst_a = tf.placeholder(tf.int32,(None))

qst = tf.placeholder(tf.int32,(None,None))
msk_qst = tf.placeholder(tf.float32,(None,None))
lst_qst = tf.placeholder(tf.int32,(None,))

opt = tf.placeholder(tf.int32,(None,None))
msk_opt = tf.placeholder(tf.float32,(None,None))
lst_opt = tf.placeholder(tf.int32,(None))

dropout_rate = tf.placeholder_with_default(1.0, shape=())

y = tf.placeholder(tf.int32, (None))

probs,att_score = trainer.forward(article,msk_a,lst_a,qst,msk_qst,lst_qst,opt,msk_opt,lst_opt,dropout_rate)
one_best = tf.argmax(probs, axis=1)

label_onehot = tf.one_hot(y, 4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=label_onehot))

test_data = load_data('RACE/data/train/middle_annotate/')
test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab,sort_by_len=False)
all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y,32)

saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess,"model/model.ckpt")
    #new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('model/'))

    predicts = []
    gold = []
    step_idx = 0
    loss_acc = 0

    for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
             mb_mask3, mb_lst3, mb_y) in enumerate(all_test):
        # Evaluate on the dev set
        train_correct = 0
        [pred_this_instance, loss_this_batch, att_this_batch] = sess.run([one_best, loss, att_score],
                                                         feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                                    qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                                    opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                                    y: mb_y, dropout_rate: 1.0})

        print att_this_batch.shape
        predicts += list(pred_this_instance)
        gold += mb_y
        step_idx += 1
        loss_acc += loss_this_batch

    dev_acc = accuracy_score(gold, predicts)
    print dev_acc

