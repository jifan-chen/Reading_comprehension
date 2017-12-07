import tensorflow as tf
import AttentionLayer_tf
from Encoder_tf import RNN_encoder
import time
import logging
import utils
from sklearn.metrics import accuracy_score

from utils import *
from preprocessing import *


class Trainer():
    def __init__(self, vocab_size, embedding_size, ini_weight, hidden_dim, bidirection = False, option_number=4):
        self.option_number = option_number
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = tf.get_variable("embedding", [vocab_size + 1, embedding_size], trainable=True,
                                          initializer=tf.constant_initializer(ini_weight))
        self.weight_mtx_dim = hidden_dim * 2 if bidirection else hidden_dim
        self.p2qa_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim )
        self.option_dot_product = AttentionLayer_tf.DotProductAttention(self.weight_mtx_dim )
        self.passage_encoder = RNN_encoder(hidden_size,'passage_encoder',bidirection)
        self.question_encoder = RNN_encoder(hidden_size,'question_encoder',bidirection)
        self.option_encoder = RNN_encoder(hidden_size,'option_encoder',bidirection)

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt):
        v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        qst_ht = tf.expand_dims(qst_ht,1)

        qstopt_ht = opt_ht + qst_ht
        #qstopt_ht = opt_ht

        #qst_ht = qst_ht.repeat(1,4,1)
        #qstopt_ht = torch.cat((opt_ht,qst_ht),dim=2)
        p2qa_align = self.p2qa_attention.score(encoded_passage,qstopt_ht)   # batch x question_num x sentence_len

        msk_p = tf.squeeze(msk_p)
        msk_p = tf.expand_dims(msk_p,1)
        p2qa_align = p2qa_align * msk_p

        #print p2qa_align.sum(dim=2)
        #print encoded_passage
        p_expectation = tf.matmul(p2qa_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim
        #print p_expectation

        o2p_align = self.option_dot_product.score(p_expectation)

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
    dev_x1, dev_x2, dev_x3, dev_x4, dev_y = convert2index(dev_data, vocab,sort_by_len=False)
    test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab,sort_by_len=False)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_y, 32)
    all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y,32)
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_y,32)

    print '-'*20,'Done','-'*20

    embedding_size = 100
    hidden_size = 128
    pre_embedding = load_pretrained_embedding('RACE/glove.6B/glove.6B.100d.txt')
    init_embedding = init_embedding_matrix(vocab,pre_embedding,embedding_size)

    trainer = Trainer(vocab_size=vocab_len,embedding_size=embedding_size,ini_weight=init_embedding,hidden_dim=hidden_size,bidirection=False)

    article = tf.placeholder(tf.int32,(None,None))
    msk_a = tf.placeholder(tf.float32,(None,None,1))
    lst_a = tf.placeholder(tf.int32,(None))

    qst = tf.placeholder(tf.int32,(None,None))
    msk_qst = tf.placeholder(tf.float32,(None,None,1))
    lst_qst = tf.placeholder(tf.int32,(None,))

    opt = tf.placeholder(tf.int32,(None,None))
    msk_opt = tf.placeholder(tf.float32,(None,None,1))
    lst_opt = tf.placeholder(tf.int32,(None))

    y = tf.placeholder(tf.int32, (None))

    probs = trainer.forward(article,msk_a,lst_a,qst,msk_qst,lst_qst,opt,msk_opt,lst_opt)
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
    debug_output = sess.run(loss, {article: all_test[0][0], msk_a: all_test[0][1], lst_a: all_test[0][2],
                                    qst: all_test[0][3], msk_qst: all_test[0][4], lst_qst: all_test[0][5],
                                    opt: all_test[0][6], msk_opt: all_test[0][7], lst_opt: all_test[0][8],
                                    y:all_test[0][9]})
    print debug_output

    print tf.trainable_variables()
    num_epoch = 50

    decay_steps = 10000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.1
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdagradOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
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
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)

        for epoch in range(num_epoch):
            step_idx = 0
            loss_acc = 0
            start_time = time.time()

            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_y) in enumerate(all_train):

                [_,loss_this_batch,summary] = sess.run([train_op, loss, merged],
                                              feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                        qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                        opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3, y:mb_y})

                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_acc += loss_this_batch

                if it % 100 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    print '-' * 10, 'Epoch ', epoch, '-' * 10, "Average Loss batch " + repr(it) + ":", round(
                        loss_acc / step_idx, 3), 'Elapsed time:', round(elapsed, 2)
                    start_time = time.time()
                    step_idx = 0
                    loss_acc = 0


            print '-' * 20, 'testing', '-' * 20
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
                                                                    opt: mb_x3, msk_opt: mb_mask3,lst_opt: mb_lst3, y: mb_y})

                predicts += list(pred_this_instance)
                gold += mb_y
                step_idx += 1
                loss_acc += loss_this_batch

            print '-' * 10, 'Test Loss ', '-' * 10, repr(loss_acc / step_idx)
            print '-' * 10,'Test Accuracy:','-' * 10, accuracy_score(gold, predicts)
