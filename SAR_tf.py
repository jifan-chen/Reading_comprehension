import tensorflow as tf
import AttentionLayer_tf
import time
from Encoder_tf import RNN_encoder
import logging
import utils
import argparse
from sklearn.metrics import accuracy_score

from utils import *
from preprocessing import *


class Trainer():
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

        self.Wa = tf.get_variable('Wa',[self.weight_mtx_dim,1],initializer=tf.contrib.layers.xavier_initializer())

        self.p2qa_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim, 'W_p2qa')
        self.p2opt_attention = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim, 'W_q2opt')

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt):
        v_passage = tf.nn.embedding_lookup(self.embedding_layer, passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer, qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer, opt)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p)
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        opt_ht = tf.reshape(opt_ht, [tf.shape(p_ht)[0], -1, self.weight_mtx_dim])

        # standard SAR

        p2q_align = self.p2q_attention.score(encoded_passage, qst_ht)  # batch x sentence_len

        #p2q_align = tf.expand_dims(p2q_align,axis=2)
        p2q_align = p2q_align * tf.squeeze(msk_p)
        p2q_align = p2q_align / tf.expand_dims(tf.reduce_sum(p2q_align, axis=1),axis=1)

        p_expectation = tf.reduce_sum(tf.expand_dims(p2q_align,axis=2) * encoded_passage, axis=1)  # encoded_passage: batch x sentence_len x emb_dim
        # print p_expectation

        o2p_align = self.o2p_attention.score(opt_ht,p_expectation)
        #o2p_align = tf.matmul(p_expectation,self.Wa)
        return o2p_align


def gen_examples(x1, x2, x3, x4, y, batch_size, opt_num, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * opt_num + k] for t in minibatch for k in range(opt_num)]
        mb_x4 = [x4[t * opt_num + k] for t in minibatch for k in range(opt_num)]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1, mb_lst1 = prepare_data(mb_x1)
        mb_x2, mb_mask2, mb_lst2 = prepare_data(mb_x2)
        mb_x3, mb_mask3, mb_lst3 = prepare_data(mb_x3)
        # mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3, mb_y))
    return all_ex

def test_model(data):
    predicts = []
    gold = []
    step_idx = 0
    loss_acc = 0

    for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
             mb_mask3, mb_lst3, mb_y) in enumerate(data):
        # Evaluate on the dev set
        train_correct = 0
        [pred_this_instance, loss_this_batch] = sess.run([one_best, loss_],
                                                         feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                                    qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                                    opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                                    y: mb_y, dropout_rnn_in: 1.0, dropout_rnn_out:1.0})

        predicts += list(pred_this_instance)
        gold += mb_y
        step_idx += 1
        loss_acc += loss_this_batch

    acc = accuracy_score(gold, predicts)

    return acc, (loss_acc/step_idx)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-train", type=str, help="path to training data")
    arg_parser.add_argument("-dev", type=str, help="path to dev data")
    arg_parser.add_argument("-test", type=str, help="path to test data")
    arg_parser.add_argument("-best_model",type=str,help='path to save the best model')
    arg_parser.add_argument("-current_model", type=str, help='path to save the current model')
    arg_parser.add_argument("-dropout_in",type=float,help='keep probability of the embedding')
    arg_parser.add_argument("-dropout_out",type=float,help='keep probability of the rnn output')

    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info('-' * 20 + 'loading training data and vocabulary' + '-' * 20)
    batch_size = 32
    train_opt_num = 2
    test_opt_num = 4
    # data loaded order: doc, question, option, Qst+Opt, Answer
    train_data = load_data(args.train)
    dev_data = load_data(args.dev)
    test_data = load_data(args.test)

    vocab = load_vocab('RACE/dict.pkl')
    #vocab = utils.build_dict(train_data[0] + train_data[1] + train_data[2])
    vocab_len = len(vocab.keys())
    embedding_size = 100
    hidden_size = 128
    init_embedding = gen_embeddings(vocab, embedding_size, 'RACE/glove.6B/glove.6B.100d.txt')

    print vocab_len

    train_x1, train_x2, train_x3, train_x4, train_y = convert2index(train_data, vocab, sort_by_len=False,opt_num=2)
    dev_x1, dev_x2, dev_x3, dev_x4, dev_y = convert2index(dev_data, vocab, sort_by_len=False,opt_num=4)
    test_x1, test_x2, test_x3, test_x4, test_y = convert2index(test_data, vocab, sort_by_len=False,opt_num=4)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_y, 32,opt_num=2)
    all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_y, 32,opt_num=4)
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_y, 32,opt_num=4)

    logging.info('-' * 20 + 'Done' + '-' * 20)

    article = tf.placeholder(tf.int32, (None, None))
    msk_a = tf.placeholder(tf.float32, (None, None))
    lst_a = tf.placeholder(tf.int32, (None))

    qst = tf.placeholder(tf.int32, (None, None))
    msk_qst = tf.placeholder(tf.float32, (None, None))
    lst_qst = tf.placeholder(tf.int32, (None,))

    opt = tf.placeholder(tf.int32, (None, None))
    msk_opt = tf.placeholder(tf.float32, (None, None))
    lst_opt = tf.placeholder(tf.int32, (None))

    #dropout_rnn_out = tf.placeholder_with_default(0.5, shape=())
    #dropout_rnn_in = tf.placeholder_with_default(0.5, shape=())
    dropout_rnn_in = tf.placeholder(tf.float32,())
    dropout_rnn_out = tf.placeholder(tf.float32, ())

    y = tf.placeholder(tf.int32, (None))

    trainer = Trainer(vocab_size=vocab_len, embedding_size=embedding_size, ini_weight=init_embedding,
                      dropout_rnn_in=dropout_rnn_in, dropout_rnn_out=dropout_rnn_out, hidden_dim=hidden_size, bidirection=True)
    probs = trainer.forward(article, msk_a, lst_a, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt)

    one_best = tf.argmax(probs, axis=1)

    label_onehot = tf.one_hot(y, train_opt_num)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=label_onehot))

    label_onehot_ = tf.one_hot(y,test_opt_num)
    loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=label_onehot_))
    # debug informaiton
    #sess = tf.Session()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #debug_output = sess.run(loss, {article: all_test[0][0], msk_a: all_test[0][1], lst_a: all_test[0][2],
    #                               qst: all_test[0][3], msk_qst: all_test[0][4], lst_qst: all_test[0][5],
    #                               opt: all_test[0][6], msk_opt: all_test[0][7], lst_opt: all_test[0][8],
    #                               y: all_test[0][9]})
    #print debug_output

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
    optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    init = tf.global_variables_initializer()
    num_epochs = 50
    merged = tf.summary.merge_all()  # merge all the tensorboard variables

    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        #train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        saver = tf.train.Saver()
        #tf.set_random_seed(2017)
        # Initialize variables
        sess.run(init)
        best_acc = 0
        nupdate = 0
        for epoch in range(num_epoch):
            step_idx = 0
            loss_acc = 0
            start_time = time.time()
            np.random.shuffle(all_train)

            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_y) in enumerate(all_train):

                [_, loss_this_batch, summary] = sess.run([train_op, loss, merged],
                                                         feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                                    qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                                    opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                                    y: mb_y, dropout_rnn_in:args.dropout_in,
                                                                    dropout_rnn_out:args.dropout_out})

                step_idx += 1
                loss_acc += loss_this_batch
                nupdate += 1

                if it % 100 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    logging.info(
                        '-' * 10 + 'Epoch ' + str(epoch) + '-' * 10 + "Average Loss batch " + repr(it) + ":" + str(
                            round(loss_acc / step_idx, 3)) + 'Elapsed time:' + str(round(elapsed, 2)))
                    start_time = time.time()
                    step_idx = 0
                    loss_acc = 0

                if nupdate % 800 == 0:
                    logging.info('-' * 20 + 'Testing on Dev' + '-' * 20)
                    dev_acc, dev_loss = test_model(all_dev)
                    logging.info('-' * 10 + 'Dev Accuracy:' + '-' * 10 + str(dev_acc))
                    logging.info('-' * 10 + 'Dev Loss ' + '-' * 10 + repr(dev_loss))
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        logging.info('-' * 10 + 'Best Dev Accuracy:' + '-' * 10 + str(best_acc))
                        # logging.info('-' * 10 + 'Saving best model ' + '-'*10)
                        logging.info('-' * 20 + 'Testing on best model' + '-' * 20)
                        saver.save(sess, args.best_model)
                        test_acc, test_loss = test_model(all_test)
                        logging.info('-' * 10 + 'Test Accuracy:' + '-' * 10 + str(test_acc))
                        logging.info('-' * 10 + 'Test loss:' + '-' * 10 + str(test_loss))
                    saver.save(sess, args.current_model)
