import tensorflow as tf
import AttentionLayer_tf
from Encoder_tf import RNN_encoder
import time
import logging
import utils
from sklearn.metrics import accuracy_score

from utils import *
from preprocessing import *

def multihop_inference():
    pass

class Trainer():
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, bidirection = False, option_number=4):
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = tf.get_variable("embedding", [vocab_size + 2, embedding_size], trainable=True,
                                          initializer=tf.constant_initializer(ini_weight))
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.m1_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim, 'W_m1')
        self.m2_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim, 'W_m2')
        self.m3_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim, 'W_m3')
        #self.option_dot_product = AttentionLayer_tf.DotProductAttention(self.weight_mtx_dim,'W_dotprod')
        self.p2opt_attention = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_p2opt')
        self.opt2p_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_opt2p')
        self.q2opt_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_q2opt')
        self.p2opt_dot = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_p2opt_dot')

        self.passage_encoder = RNN_encoder(hidden_dim,'passage_encoder',bidirection,keep_prob=0.5,reuse=tf.AUTO_REUSE)
        self.question_encoder = RNN_encoder(hidden_dim,'question_encoder',bidirection,keep_prob=0.5)
        self.option_encoder = RNN_encoder(hidden_dim,'option_encoder',bidirection,keep_prob=0.5)
        self.gated_encoder1 = RNN_encoder(hidden_dim,'g1_encoder',bidirection,keep_prob=0.5,reuse=tf.AUTO_REUSE)
        self.gated_encoder2 = RNN_encoder(hidden_dim,'g2_encoder',bidirection,keep_prob=0.5)

        self.psg_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'passage')
        self.qst_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'question')
        self.opt_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'option')

        self.gated_att1 = AttentionLayer_tf.GatedAttention(self.hidden_dim, self.weight_mtx_dim, 'gated1')
        self.gated_att2 = AttentionLayer_tf.GatedAttention(self.hidden_dim, self.weight_mtx_dim, 'gated2')

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt,dropout_rate):
        v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        #attended_passage = self.psg_selfatt.apply_self_attention(encoded_passage)
        #encoded_passage = attended_passage
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        #encoded_question = self.qst_selfatt.apply_self_attention(encoded_question)
        #qst_ht += tf.reduce_sum(encoded_question,axis=1)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)
        #encoded_option = self.opt_selfatt.apply_self_attention(encoded_option)
        #opt_ht += tf.reduce_sum(encoded_option,axis=1)


        gated1 = self.gated_att1.apply_attention(encoded_option,encoded_passage,msk_opt)
        ght1,encoded_g1 = self.gated_encoder1.encode(gated1,lst_p)
        gated2 = self.gated_att1.apply_attention(encoded_option, encoded_g1, msk_opt)
        #ght2, encoded_g2 = self.gated_encoder1.encode(gated2, lst_p)
        #gated3 = self.gated_att1.apply_attention(encoded_option, encoded_g2, msk_qst)


        p2opt_align = self.p2opt_attention.score(gated2, opt_ht)  # p2opt_align: batch x question_num x sentence_len
        p2opt_align = p2opt_align * msk_p

        p2opt_align = p2opt_align / tf.expand_dims(tf.reduce_sum(p2opt_align, axis=1), axis=1)

        popt_expectation = tf.reduce_sum(tf.expand_dims(p2opt_align, axis=2) * encoded_passage, axis=1)
        popt_expectation = tf.reshape(popt_expectation,[-1, self.option_number, self.weight_mtx_dim])
        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht

        probs = self.p2opt_dot.score(qstopt_ht,popt_expectation)
        return probs

        #popt_expectation = tf.matmul(p2opt_align, encoded_passage)

        '''
        msk_p = tf.expand_dims(msk_p, 1)
        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht
        m1_align = self.m1_attention.score(encoded_passage, qstopt_ht)  # p2qa_align: batch x question_num x sentence_len
        m1_align = m1_align * msk_p
        m1_align = m1_align / tf.expand_dims(tf.reduce_sum(m1_align, axis=2), axis=2)
        m1 = tf.matmul(m1_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim

        m2_align = self.m2_attention.score(encoded_passage, qstopt_ht + m1)
        m2_align = m2_align * msk_p
        m2_align = m2_align / tf.expand_dims(tf.reduce_sum(m2_align, axis=2), axis=2)
        m2 = tf.matmul(m2_align,encoded_passage)

        m3_align = self.m3_attention.score(encoded_passage, qstopt_ht + m2)
        m3_align = m3_align * msk_p
        m3_align = m3_align / tf.expand_dims(tf.reduce_sum(m3_align, axis=2), axis=2)
        m3 = tf.matmul(m3_align, encoded_passage)

        m = m1 + m2 + m3
        expectation = m
        '''
        o2q_align = self.p2opt_dot.score(expectation,qstopt_ht)

        return o2q_align

def convert2index(examples, word_dict,
                  sort_by_len=True, verbose=True, concat=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_x3 = []
    in_x4 = []
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 0 for w in st]
        return seq

    for idx, (q, a) in enumerate(zip(examples[1], examples[4])):
        q_words = q.split(' ')
        assert 0 <= a <= 3
        seq2 = get_vector(q_words)
        if (len(seq2) > 0):
            in_x2 += [seq2]
            option_seq = []
            qsop_seq  = []
            psg_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * 4]
                else:
                    op = examples[2][i + idx * 4]
                    qsop = examples[3][i + idx*4]
                    psg = examples[0][i+idx*4]
                op = op.split(' ')
                qsop = qsop.split(' ')
                psg = psg.split(' ')
                option = get_vector(op)
                question_option = get_vector(qsop)
                passage = get_vector(psg)
                assert len(option) > 0
                option_seq += [option]
                qsop_seq += [question_option]
                psg_seq += [passage]

            in_x1 += [psg_seq]
            in_x3 += [option_seq]
            in_x4 += [qsop_seq]
            in_y.append(a)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[1])))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        #print sorted_index
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        in_x3 = [in_x3[i] for i in sorted_index]
        in_x4 = [in_x4[i] for i in sorted_index]
    new_in_x1 = []
    new_in_x3 = []
    new_in_x4 = []
    for i,j,k in zip(in_x1,in_x3,in_x4):
        #print i
        new_in_x1 += i
        new_in_x3 += j
        new_in_x4 += k
    #print new_in_x3
    return new_in_x1, in_x2, new_in_x3, new_in_x4, in_y

def gen_examples(x1, x2, x3, x4, y ,batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x2), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x4 = [x4[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1, mb_lst1 = prepare_data(mb_x1)
        mb_x2, mb_mask2, mb_lst2 = prepare_data(mb_x2)
        mb_x3, mb_mask3, mb_lst3 = prepare_data(mb_x3)
        #mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3 ,mb_y))
    return all_ex

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('-'*20 + 'loading training data and vocabulary' + '-'*20)
    vocab = load_vocab('RACE/dict.pkl')
    vocab_len = len(vocab.keys())
    print vocab_len
    batch_size = 32

    # data loaded order: doc, question, option, Qst+Opt, Answer
    train_data= load_data('fact_questions/train/middle')
    dev_data = load_data('fact_questions/dev/middle')
    test_data = load_data('RACE/data/test/middle/')

    train_x1, train_x2, train_x3, train_x4, train_y = convert2index(train_data, vocab,sort_by_len=False)
    dev_x1, dev_x2, dev_x3, dev_x4, dev_y = convert2index(dev_data, vocab,sort_by_len=False)
    test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab,sort_by_len=False)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_y, 32)
    all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y,32)
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_y,32)

    logging.info('-'*20 +'Done' + '-'*20)

    embedding_size = 100
    hidden_size = 128
    #pre_embedding = load_pretrained_embedding('RACE/glove.6B/glove.6B.100d.txt')
    #init_embedding = init_embedding_matrix(vocab,pre_embedding,embedding_size)
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

    probs = trainer.forward(article,msk_a,lst_a,qst,msk_qst,lst_qst,opt,msk_opt,lst_opt,dropout_rate)
    one_best = tf.argmax(probs, axis=1)

    label_onehot = tf.one_hot(y, 4)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=label_onehot))
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=probs,labels=label_onehot)
    #loss = tf.reduce_mean(tf.negative(tf.log(tf.reduce_sum(probs * label_onehot, axis=1))))

    # debug informaiton
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    #print all_test[0][0].shape, all_test[0][1].shape, all_test[0][2]
    #print all_test[0][9]

    debug_output = sess.run(probs, {article: all_test[0][0], msk_a: all_test[0][1], lst_a: all_test[0][2],
                                    qst: all_test[0][3], msk_qst: all_test[0][4], lst_qst: all_test[0][5],
                                    opt: all_test[0][6], msk_opt: all_test[0][7], lst_opt: all_test[0][8],
                                    y:all_test[0][9],dropout_rate:0.5})
    print 'debug output:',debug_output.shape

    print tf.trainable_variables()
    num_epoch = 50

    decay_steps = 10000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.3
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    def clip_not_none(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -10, 10)
    capped_gvs = [(clip_not_none(grad), var) for grad, var in grads]
    apply_gradient_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 50
    merged = tf.summary.merge_all()  # merge all the tensorboard variables



    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        saver = tf.train.Saver()
        best_acc = 0
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)

        nupdate = 0
        for epoch in range(num_epoch):
            step_idx = 0
            loss_acc = 0
            start_time = time.time()
            np.random.shuffle(all_train)
            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_y) in enumerate(all_train):

                [_,loss_this_batch] = sess.run([train_op, loss],
                                              feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                        qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                        opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3, y:mb_y,dropout_rate:0.5})

                step_idx += 1
                loss_acc += loss_this_batch
                nupdate += 1
                if it % 100 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    logging.info('-' * 10 + 'Epoch ' + str(epoch) + '-' * 10 + "Average Loss batch " + repr(it) + ":" + str(round(
                        loss_acc / step_idx, 3)) + 'Elapsed time:' + str(round(elapsed, 2)))
                    start_time = time.time()
                    step_idx = 0
                    loss_acc = 0

                if nupdate % 1000 == 0:
                    logging.info('-' * 20 +'testing' + '-' * 20)
                    predicts = []
                    gold = []
                    step_idx = 0
                    loss_acc = 0


                    for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                             mb_mask3, mb_lst3, mb_y) in enumerate(all_dev):
                        # Evaluate on the dev set
                        train_correct = 0
                        [pred_this_instance, loss_this_batch] = sess.run([one_best, loss],
                                                                         feed_dict={article: mb_x1, msk_a: mb_mask1,lst_a: mb_lst1,
                                                                            qst: mb_x2, msk_qst: mb_mask2,lst_qst: mb_lst2,
                                                                            opt: mb_x3, msk_opt: mb_mask3,lst_opt: mb_lst3, y: mb_y, dropout_rate:1.0})

                        predicts += list(pred_this_instance)
                        gold += mb_y
                        step_idx += 1
                        loss_acc += loss_this_batch

                    dev_acc = accuracy_score(gold, predicts)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        logging.info('-' * 10 + 'Saving best model ' + '-'*10)
                        saver.save(sess,"model/best_model")
                    saver.save(sess,"model/current_model")
                    logging.info('-' * 10 + 'Test Loss ' + '-' * 10 + repr(loss_acc / step_idx))
                    logging.info('-' * 10 + 'Test Accuracy:' + '-' * 10 + str(accuracy_score(gold, predicts)))
                    logging.info('-' * 10 + 'Best Accuracy:' + '-'*10 + str(best_acc))
