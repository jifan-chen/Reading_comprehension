import tensorflow as tf
import AttentionLayer_tf
from Encoder_tf import RNN_encoder
import time
import logging
import utils
from sklearn.metrics import accuracy_score
from rouge import Rouge

from utils import *
from preprocessing import *

rouge = Rouge()


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
        self.e2opt_attention = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_e2opt')
        self.e2opt_dot1 = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_e2opt_dot1')
        self.e2opt_dot2 = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_e2opt_dot2')

        self.passage_encoder = RNN_encoder(hidden_dim,'passage_encoder',bidirection,keep_prob=0.8)
        self.question_encoder = RNN_encoder(hidden_dim,'question_encoder',bidirection,keep_prob=0.8)
        self.option_encoder = RNN_encoder(hidden_dim,'option_encoder',bidirection,keep_prob=0.8)
        self.evidence_encoder = RNN_encoder(hidden_dim,'evidence_encoder',bidirection,keep_prob=0.8)

        self.psg_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'passage')
        self.evd_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'evidence')

    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt,
                evd, msk_evd, lst_evd, evd_score, dropout_rate):

        #v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)
        v_evd = tf.nn.embedding_lookup(self.embedding_layer,evd)

        #p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        #attended_passage = self.psg_selfatt.apply_self_attention(encoded_passage)
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        #encoded_question = self.qst_selfatt.apply_self_attention(encoded_question)
        #qst_ht += tf.reduce_mean(encoded_question,axis=1)
        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)
        #encoded_option = self.opt_selfatt.apply_self_attention(encoded_option)
        #opt_ht += tf.reduce_mean(encoded_option,axis=1)

        evd_ht, encoded_evidence = self.evidence_encoder.encode(v_evd, lst_evd)
        #encoded_evidence = self.evd_selfatt.apply_self_attention(encoded_evidence)
        e2opt_att = self.e2opt_attention.score(encoded_evidence,opt_ht)
        evd_ht = tf.reduce_sum(tf.expand_dims(e2opt_att,axis=2) * encoded_evidence,axis=1)
        evd_score = tf.reshape(evd_score, [-1, self.option_number, 1])
        evd_score = evd_score / tf.expand_dims(tf.reduce_sum(evd_score, axis=1), axis=1)
        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        evd_ht = tf.reshape(evd_ht, [-1, self.option_number, self.weight_mtx_dim])
        evd_ht = evd_ht * evd_score
        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht

        probs_qstopt = self.e2opt_dot1.score(evd_ht,qstopt_ht)
        probs_opt = self.e2opt_dot2.score(evd_ht,opt_ht)
        return probs_qstopt + probs_opt

        evd_score = tf.reshape(evd_score,[-1,self.option_number,1])
        evd_score = evd_score / tf.expand_dims(tf.reduce_sum(evd_score,axis=1),axis=1)
        sum_evd = tf.reduce_sum(evd_ht * evd_score,axis=1)

        #p2opt_align = self.p2opt_attention.score(encoded_passage, opt_ht)  # p2opt_align: batch x question_num x sentence_len
        #p2opt_align = tf.reduce_sum(p2opt_align)
        #p2opt_align = p2opt_align * msk_p

        #p2opt_align = p2opt_align / tf.expand_dims(tf.reduce_sum(p2opt_align, axis=1), axis=1)

        #popt_expectation = tf.reduce_sum(tf.expand_dims(p2opt_align, axis=2) * encoded_passage, axis=1)
        popt_expectation = sum_evd
        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht

        probs = self.opt2p_attention.score(qstopt_ht,popt_expectation)
        return probs
        '''
        #popt_expectation = tf.matmul(p2opt_align, encoded_passage)

        msk_p = tf.expand_dims(msk_p, 1)
        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht
        p2qa_align = self.p2qa_attention.score(encoded_passage,qstopt_ht)  # p2qa_align: batch x question_num x sentence_len
        p2qa_align = p2qa_align * msk_p
        p2qa_align = p2qa_align / tf.expand_dims(tf.reduce_sum(p2qa_align, axis=2), axis=2)
        pqa_expectation = tf.matmul(p2qa_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim

        #expectation = pqa_expectation + popt_expectation
        expectation = pqa_expectation
        #expectation = popt_expectation
        #o2p_align = self.option_dot_product.score(pqa_expectation)
        o2q_align = self.q2opt_attention.score(expectation,tf.squeeze(qst_ht))

        return o2q_align,p2opt_align
        '''


def extract_evidence(psg, opts, qst_opts):
    # print opts
    tkps = tokenize.sent_tokenize(psg)
    evidences = []
    best_scores = []
    for opt, qst_opt in zip(opts, qst_opts):
        best = 0
        # print opt
        for s in tkps:
            score1 = rouge.get_scores([opt], [s])
            rouge1 = score1[0]['rouge-1']['f']

            score2 = rouge.get_scores([qst_opt], [s])
            rouge2 = score2[0]['rouge-1']['f']

            rouge_combine = rouge1 + rouge2
            if rouge_combine >= best:
                e = s
                best = rouge_combine
        best_scores.append(best + 1e-5)
        evidences.append(e)
    # print evidences
    # print best_scores
    return evidences, best_scores

def load_data(in_file, max_example=None, relabeling=True):

    documents = []
    questions = []
    answers = []
    options = []
    qs_op = []
    question_belong = []
    evidences = []
    best_scores = []
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

    count = 0
    for inf in files:
        try:
            obj = json.load(open(inf, "r"))
        except ValueError:
            print inf
            continue

        for i, q in enumerate(obj["questions"]):
            question_belong += [inf]
            documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            for j in range(4):
                if '_' in q.split():
                    qs_op += [q.replace('_', obj['options'][i][j])]
                else:
                    qs_op += [q + ' ' + obj['options'][i][j]]
            options += obj["options"][i]
            evds, scores = extract_evidence(obj['article'], obj["options"][i], qs_op[count * 4:(count + 1) * 4])
            evidences += evds
            best_scores += scores

            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
            count += 1
        if (max_example is not None) and (num_examples >= max_example):
            break

    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list

    documents = clean(documents)
    questions = clean(questions)
    options = clean(options)
    evidences = clean(evidences)
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options,qs_op,evidences,best_scores,answers)

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
    in_x5 = []
    in_x6 = examples[5]
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 0 for w in st]
        return seq

    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[6])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        assert 0 <= a <= 3
        seq1 = get_vector(d_words)
        seq2 = get_vector(q_words)
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
            option_seq = []
            qsop_seq  = []
            evd_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * 4]
                else:
                    op = examples[2][i + idx * 4]
                    qsop = examples[3][i + idx*4]
                    evd = examples[4][i + idx*4]
                op = op.split(' ')
                qsop = qsop.split(' ')
                option = get_vector(op)
                question_option = get_vector(qsop)
                evidence = get_vector(evd)
                assert len(option) > 0
                option_seq += [option]
                qsop_seq += [question_option]
                evd_seq += [evidence]
            in_x3 += [option_seq]
            in_x4 += [qsop_seq]
            in_x5 += [evd_seq]
            in_y.append(a)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

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
        in_x5 = [in_x5[i] for i in sorted_index]
    new_in_x3 = []
    new_in_x4 = []
    new_in_x5 = []
    for i,j,k in zip(in_x3,in_x4,in_x5):
        #print i
        new_in_x3 += i
        new_in_x4 += j
        new_in_x5 += k
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, new_in_x4, new_in_x5, in_x6, in_y

def gen_examples(x1, x2, x3, x4, x5, x6, y ,batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x5 = [x5[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x6 = [x6[t * 4 + k] for t in minibatch for k in range(4)]

        #mb_x4 = [x4[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1, mb_lst1 = prepare_data(mb_x1)
        mb_x2, mb_mask2, mb_lst2 = prepare_data(mb_x2)
        mb_x3, mb_mask3, mb_lst3 = prepare_data(mb_x3)
        mb_x5, mb_mask5, mb_lst5 = prepare_data(mb_x5)
        #mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3,
                       mb_x5, mb_mask5, mb_lst5, mb_x6, mb_y))
    return all_ex

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('-'*20 + 'loading training data and vocabulary' + '-'*20)
    vocab = load_vocab('RACE/dict.pkl')
    vocab_len = len(vocab.keys())
    print vocab_len
    batch_size = 32

    # data loaded order: doc, question, option, Qst+Opt, Answer
    train_data= load_data('none_fact_questions/train/middle')
    dev_data = load_data('none_fact_questions/test/middle')
    #test_data = load_data('RACE/data/test/middle/')

    train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_y = convert2index(train_data, vocab,sort_by_len=False)
    dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_x6, dev_y = convert2index(dev_data, vocab,sort_by_len=False)

    #test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_y = convert2index(test_data, vocab,sort_by_len=False)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_x5,train_x6,train_y, 32)
    #all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_y,32)
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_x6, dev_y,32)

    logging.info('-'*20 +'Done' + '-'*20)

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

    evd = tf.placeholder(tf.int32,(None,None))
    msk_evd = tf.placeholder(tf.float32,(None,None))
    lst_evd = tf.placeholder(tf.int32,(None))

    evd_score = tf.placeholder(tf.float32,(None))

    dropout_rate = tf.placeholder_with_default(1.0, shape=())

    y = tf.placeholder(tf.int32, (None))

    probs = trainer.forward(article,msk_a,lst_a,qst,msk_qst,lst_qst,opt,msk_opt,
                            lst_opt,evd,msk_evd,lst_evd,evd_score,dropout_rate)
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
    debug_output = sess.run(probs, {article: all_dev[0][0], msk_a: all_dev[0][1], lst_a: all_dev[0][2],
                                    qst: all_dev[0][3], msk_qst: all_dev[0][4], lst_qst: all_dev[0][5],
                                    opt: all_dev[0][6], msk_opt: all_dev[0][7], lst_opt: all_dev[0][8],
                                    evd: all_dev[0][9], msk_evd: all_dev[0][10], lst_evd: all_dev[0][11],
                                    evd_score:all_dev[0][12], y:all_dev[0][13], dropout_rate:0.5})
    print 'debug output:',debug_output

    print tf.trainable_variables()
    num_epoch = 50

    decay_steps = 1000
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
        saver = tf.train.Saver()
        best_acc = 0
        #tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)

        for epoch in range(num_epoch):
            step_idx = 0
            loss_acc = 0
            start_time = time.time()
            np.random.shuffle(all_train)
            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_x4, mb_mask4, mb_lst4, mb_s, mb_y) in enumerate(all_train):

                [_,loss_this_batch] = sess.run([train_op, loss],
                                              feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                        qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                        opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                        evd: mb_x4, msk_evd: mb_mask4, lst_evd: mb_lst4,
                                        evd_score:mb_s, y:mb_y,dropout_rate:0.5})

                step_idx += 1
                loss_acc += loss_this_batch

                if it % 100 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    logging.info('-' * 10 + 'Epoch ' + str(epoch) + '-' * 10 + "Average Loss batch " + repr(it) + ":" + str(round(
                        loss_acc / step_idx, 3)) + 'Elapsed time:' + str(round(elapsed, 2)))
                    start_time = time.time()
                    step_idx = 0
                    loss_acc = 0


            logging.info('-' * 20 +'testing' + '-' * 20)
            predicts = []
            gold = []
            step_idx = 0
            loss_acc = 0

            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_x4, mb_mask4, mb_lst4, mb_s, mb_y) in enumerate(all_dev):
                # Evaluate on the dev set
                train_correct = 0
                [pred_this_instance, loss_this_batch] = \
                    sess.run([one_best, loss],feed_dict={article: mb_x1, msk_a: mb_mask1,lst_a: mb_lst1,
                        qst: mb_x2, msk_qst: mb_mask2,lst_qst: mb_lst2,
                        opt: mb_x3, msk_opt: mb_mask3,lst_opt: mb_lst3,
                        evd: mb_x4, msk_evd: mb_mask4, lst_evd: mb_lst4,
                        evd_score: mb_s,y: mb_y, dropout_rate:1.0})

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
