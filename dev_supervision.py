import tensorflow as tf
import AttentionLayer_tf
from Encoder_tf import RNN_encoder
import time
import argparse
from nltk.corpus import stopwords
import logging
import utils
from sklearn.metrics import accuracy_score
from rouge import Rouge
from utils import *

rouge = Rouge()
stop_words = list(stopwords.words('english'))
stop_words += [',','.','?','!']
stop_words = set(stop_words)
print stop_words

def cross_entropy(p,q):
    '''
    :param p: batch x len
    :param q: batch x len
    :return: KL(p||q) + H(p)
    '''
    return tf.reduce_sum(- p * tf.log(q),axis=1) * 1e-3

def negative_likelihood(p, q ,axis=2):
    '''
    :param p: batch x len
    :param q: batch x len
    :return: KL(p||q) + H(p)
    '''
    #q += 1e-5
    return -tf.log(tf.reduce_sum(q * p,axis=axis))

class Trainer():
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
        self.p2q_attentionc1 = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_p2qc1')
        self.o2p_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim, 'W_o2p')
        self.p2q_attentionc2 = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_p2qc2')

        self.passage_encoder = RNN_encoder(hidden_dim, 'passage_encoder', bidirection,
                                           input_keep_prob=self.dropout_rnn_in,
                                           output_keep_prob=self.dropout_rnn_out, reuse=tf.AUTO_REUSE)
        self.question_encoder = RNN_encoder(hidden_dim, 'question_encoder', bidirection,
                                            input_keep_prob=self.dropout_rnn_in,
                                            output_keep_prob=self.dropout_rnn_out)
        self.question_encoder_ = RNN_encoder(hidden_dim,'question_encoder_',bidirection,
                                             input_keep_prob=self.dropout_rnn_in,
                                             output_keep_prob=self.dropout_rnn_out)
        self.option_encoder = RNN_encoder(hidden_dim, 'option_encoder', bidirection,
                                          input_keep_prob=self.dropout_rnn_in,
                                          output_keep_prob=self.dropout_rnn_out)


        self.W1 = tf.get_variable('W1',[self.weight_mtx_dim,self.weight_mtx_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W1_ = tf.get_variable('W1_',[self.weight_mtx_dim,self.weight_mtx_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())



    def forward(self, passage, msk_p, lst_p, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt, att_msk, answer, if_test):

        v_passage = tf.nn.embedding_lookup(self.embedding_layer,passage)
        v_qst = tf.nn.embedding_lookup(self.embedding_layer,qst)
        v_opt = tf.nn.embedding_lookup(self.embedding_layer,opt)

        p_ht, encoded_passage = self.passage_encoder.encode(v_passage, lst_p )
        qst_ht, encoded_question = self.question_encoder.encode(v_qst, lst_qst)
        qst_ht_, encoded_question_ = self.question_encoder_.encode(v_qst,lst_qst)

        opt_ht, encoded_option = self.option_encoder.encode(v_opt, lst_opt)

        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        att_msk = tf.reshape(att_msk, [tf.shape(passage)[0],self.option_number,-1])

        def attend_by_qa(att_msk, encoded_passage, qst_ht, msk_p, answer):
            msk_p = tf.expand_dims(msk_p,1)
            qst_ht = tf.expand_dims(qst_ht, 1)
            qstopt_ht = opt_ht + qst_ht
            p2qa_align = self.p2qa_attention.score(encoded_passage,qstopt_ht)  # p2qa_align: batch x option_num x sentence_len

            nl = negative_likelihood(att_msk, p2qa_align)
            nl = nl * answer
            p2qa_align = tf.cond(if_test, lambda: p2qa_align, lambda: p2qa_align * att_msk)
            nl = tf.reduce_sum(tf.reshape(nl, [-1, self.option_number]), axis=1)
            p2qa_align = p2qa_align * msk_p
            p2qa_align = p2qa_align / tf.expand_dims(tf.reduce_sum(p2qa_align, axis=2), axis=2)

            pqa_expectation = tf.matmul(p2qa_align, encoded_passage)  # encoded_passage: batch x sentence_len x emb_dim
            expectation = pqa_expectation
            o2q_align = self.p2opt_attention.score(expectation, qstopt_ht)

            return o2q_align, nl

        def attend_by_q(self, att_msk, encoded_passage, qst_ht, msk_p, answer):
            p2q_align = self.p2q_attentionc1.score(encoded_passage, qst_ht)
            p2q_align_ = self.p2q_attentionc2.score(encoded_passage, qst_ht_)

            answer = tf.expand_dims(answer, 2)
            att_msk = tf.reduce_sum(att_msk * answer, axis=1)
            nl = negative_likelihood(att_msk, p2q_align, axis=1)

            p2q_align = tf.cond(if_test, lambda: p2q_align, lambda: p2q_align * att_msk)
            p2q_align = p2q_align * tf.squeeze(msk_p)
            p2q_align = p2q_align / tf.expand_dims(tf.reduce_sum(p2q_align, axis=1), axis=1)

            p2q_align_ = p2q_align_ * tf.squeeze(msk_p)
            p2q_align_ = p2q_align_ / tf.expand_dims(tf.reduce_sum(p2q_align_, axis=1), axis=1)

            s1 = tf.reduce_sum(tf.expand_dims(p2q_align, axis=2) * encoded_passage, axis=1)
            s1_ = tf.reduce_sum(tf.expand_dims(p2q_align_, axis=2) * encoded_passage, axis=1)

            #q2 = tf.expand_dims(s1 + qst_ht,1)
            #q2_ = s1_ + qst_ht
            #q2 = tf.nn.relu(tf.tensordot(q2, self.W1,[[2],[0]]))
            #q2_ = tf.nn.relu(tf.matmul(q2_, self.W1_))

            #p2q_align2 = tf.nn.softmax(tf.reduce_sum(q2 * encoded_passage,axis=2,keep_dims=True))
            #s2 = tf.reduce_sum(p2q_align2 * encoded_passage,axis=1)

            o2p_align = self.o2p_attention.score(opt_ht, s1 + s1_)

            return o2p_align, nl

        #return attend_by_qa(att_msk,encoded_passage,qst_ht,msk_p,answer)
        return attend_by_q(self,att_msk,encoded_passage,qst_ht,msk_p,answer)


def extract_evidence_by_ngram(psg, opts, qst_opts):
    psg = psg.split(" ")
    evidences = opts
    best_scores = [1,1,1,1]
    evd_msk = []
    idxs = []
    for qst_opt in qst_opts:
        m = np.zeros(len(psg))
        for w in qst_opt.split():
            match_index = [i for i,s in enumerate(psg) if w not in stop_words and w in s ]

            idxs += match_index

        if len(idxs) == 0:
            m[:] = 1.0
            print 'No evidences find'
            print psg
            for qst_opt in qst_opts:
                print qst_opt.split()

        for i in idxs:
            m[i] = 1
        #print m
        evd_msk.append(m)

    return evidences, best_scores, evd_msk

def extract_evidence(psg, opts, qst_opts,answer):
    # print opts
    tkps = tokenize.sent_tokenize(psg)
    evidences = []
    if_fact = False

    for i, oq in enumerate(zip(opts, qst_opts)):
        opt = oq[0]
        qst_opt = oq[1]
        best = 0

        # print opt
        for s in tkps:
            score1 = rouge.get_scores([opt], [s])
            rouge1 = score1[0]['rouge-1']['f']
            rouge1p = score1[0]['rouge-1']['p']
            if rouge1p > 0.9 and i == ord(answer) - ord('A'):
                if_fact = True

            score2 = rouge.get_scores([qst_opt], [s])
            rouge2 = score2[0]['rouge-1']['f']

            rouge_combine = rouge1 + rouge2
            if rouge_combine >= best:
                best = rouge_combine
                e = s
        evidences.append(e)

    #print evidences
    #print if_fact
    #if if_fact:
    return evidences,if_fact
    #else:
    #    return []

def generate_att_mask(psg,evidences,fact):

    masks = []
    if not fact or len(evidences) == 0:
        for i in range(4):
            masks.append(np.ones(len(psg.split())))
        return masks

    for evd in evidences:
        m = np.zeros(len(psg.split(" ")))
        index = psg.index(evd)
        _psg = psg[:index]
        _psg_list = _psg.split(" ")
        start = len(_psg_list) - 1
        psg_list = psg.split(" ")
        evd_list = evd.split(" ")
        evd_len = len(evd_list)

        m[start:start+evd_len] = 1
        masks.append(m)

    return masks


def load_data(in_file, max_example=None, gen_mask = True):

    documents = []
    questions = []
    answers = []
    options = []
    qs_op = []
    question_belong = []
    att_masks = []
    num_examples = 0

    total_num_qst = 0
    hit_num_qst = 0

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

        evidences = []
        fact = []
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
            #print generate_att_mask(obj['article'],obj["evidences"][i])
            if gen_mask:
                #att_masks.append(generate_att_mask(obj['article'],obj["evidences"][i]))
                #print generate_att_mask(obj['article'],obj["evidences"][i],obj['fact'][i])
                #print generate_att_mask(obj['article'],obj["evidences"][i],True)
                att_masks += generate_att_mask(obj['article'],obj["evidences"][i],obj['fact'][i])
                #att_masks += generate_att_mask(obj['article'],obj["evidences"][i],True)
            else:
                att_masks.append([])
            #evds,fact_this_q = extract_evidence(obj['article'], obj["options"][i], qs_op[count * 4:(count + 1) * 4],obj['answers'][i])
            #fact.append(fact_this_q)
            #if len(evds):
            #    hit_num_qst += 1

            #total_num_qst += 1
            #evidences.append(evds)
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
            count += 1
        #obj['evidences'] = evidences
        #obj['fact'] = fact
        #json.dump(obj, open(inf, "w"), indent=4)

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
    #print total_num_qst
    #print hit_num_qst
    return documents, questions, options, qs_op, att_masks, answers

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
    in_attmask = examples[4]
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 0 for w in st]
        return seq

    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[5])):
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
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * 4]
                else:
                    op = examples[2][i + idx * 4]
                    qsop = examples[3][i + idx*4]
                op = op.split(' ')
                qsop = qsop.split(' ')
                option = get_vector(op)
                question_option = get_vector(qsop)
                assert len(option) > 0
                option_seq += [option]
                qsop_seq += [question_option]
            in_x3 += [option_seq]
            in_x4 += [qsop_seq]
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
    new_in_x3 = []
    new_in_x4 = []

    for i,j in zip(in_x3,in_x4):
        new_in_x3 += i
        new_in_x4 += j

    return in_x1, in_x2, new_in_x3, new_in_x4, in_attmask, in_y

def prepare_att_mask(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x_mask = np.zeros((n_samples, max_len)).astype(np.float)
    for idx, seq in enumerate(seqs):
        x_mask[idx, :lengths[idx]] = seq
    return x_mask

def gen_examples(x1, x2, x3, x4, x_attmask, y ,batch_size, gen_mask=True):
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

        if gen_mask:
            mb_xmask = [x_attmask[t * 4 + k] for t in minibatch for k in range(4)]
            mb_xmask = prepare_att_mask(mb_xmask)
        else:
            mb_xmask = [x_attmask[t] for t in minibatch]
            mb_xmask = prepare_att_mask(mb_xmask)
        #mb_x4 = [x4[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1, mb_lst1 = prepare_data(mb_x1)
        mb_x2, mb_mask2, mb_lst2 = prepare_data(mb_x2)
        mb_x3, mb_mask3, mb_lst3 = prepare_data(mb_x3)
        #print mb_x1.shape
        #mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)

        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3, mb_mask3, mb_lst3,mb_xmask,mb_y))

    return all_ex

def test_model(data):
    predicts = []
    gold = []
    step_idx = 0
    loss_acc = 0

    for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
             mb_mask3, mb_lst3,mb_att,mb_y) in enumerate(data):
        # Evaluate on the dev set
        train_correct = 0
        [pred_this_instance, loss_this_batch] = \
            sess.run([one_best, class_loss], feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                  qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                  opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                  att_msk:mb_att, y: mb_y, dropout_rnn_in:1.0,
                                                  dropout_rnn_out:1.0, if_test:True})

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
    arg_parser.add_argument("-best_model", type=str, help='path to save the best model')
    arg_parser.add_argument("-current_model", type=str, help='path to save the current model')
    arg_parser.add_argument("-dropout_in", type=float, help='keep probability of the embedding')
    arg_parser.add_argument("-dropout_out", type=float, help='keep probability of the rnn output')
    arg_parser.add_argument("-epoch_num", type=int, help='number of training epoch', default=50)
    arg_parser.add_argument("-debug",type=bool,help='debug or not',default=False)
    arg_parser.add_argument("-continue_train",type=bool, help ='train the existing model or not',default=False)
    arg_parser.add_argument("-logging_file", type=str, help='path to the logging file', default=None)
    arg_parser.add_argument('-device', type=str, help='Which GPU to use', default="0")
    args = arg_parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.logging_file is not None:
        logging.basicConfig(filename=args.logging_file, filemode='w', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info('-' * 20 + 'loading training data and vocabulary' + '-' * 20)
    batch_size = 32

    print args.debug
    # data loaded order: doc, question, option, Qst+Opt, Answer

    if args.debug:
        vocab = load_vocab('RACE/dict.pkl')
        dev_data = load_data(args.dev)
        dev_x1, dev_x2, dev_x3, dev_x4, dev_xmask, dev_y = convert2index(dev_data, vocab,sort_by_len=False)
        all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_xmask, dev_y, 32)
        all_train = all_dev
    else:
        train_data = load_data(args.train,gen_mask=True)
        test_data = load_data(args.test,gen_mask=True)
        dev_data = load_data(args.dev,gen_mask=True)
        vocab = load_vocab('RACE/dict.pkl')
        #vocab = utils.build_dict(train_data[0] + train_data[1] + train_data[2])
        train_x1, train_x2, train_x3, train_x4, train_xmask, train_y = convert2index(train_data, vocab,sort_by_len=False)
        all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_xmask, train_y, 32)
        test_x1, test_x2, test_x3, test_x4, test_xmask, test_y = convert2index(test_data, vocab,sort_by_len=False)
        all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_xmask, test_y, 32)
        dev_x1, dev_x2, dev_x3, dev_x4, dev_xmask, dev_y = convert2index(dev_data, vocab, sort_by_len=False)
        all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_xmask, dev_y, 32)
    #vocab = load_vocab('RACE/dict.pkl')

    #vocab = utils.build_dict(dev_data[0] + dev_data[1] + dev_data[2])

    vocab_len = len(vocab.keys())
    embedding_size = 100
    hidden_size = 128
    init_embedding = gen_embeddings(vocab, embedding_size, 'RACE/glove.6B/glove.6B.100d.txt')

    print vocab_len

    logging.info('-'*20 +'Done' + '-'*20)

    article = tf.placeholder(tf.int32,(None,None))
    msk_a = tf.placeholder(tf.float32,(None,None))
    lst_a = tf.placeholder(tf.int32,(None))

    qst = tf.placeholder(tf.int32,(None,None))
    msk_qst = tf.placeholder(tf.float32,(None,None))
    lst_qst = tf.placeholder(tf.int32,(None,))

    opt = tf.placeholder(tf.int32,(None,None))
    msk_opt = tf.placeholder(tf.float32,(None,None))
    lst_opt = tf.placeholder(tf.int32,(None))

    att_msk = tf.placeholder(tf.float32,(None,None))
    if_test = tf.placeholder(tf.bool, (None))

    dropout_rnn_out = tf.placeholder_with_default(0.5, shape=())
    dropout_rnn_in = tf.placeholder_with_default(0.5, shape=())

    y = tf.placeholder(tf.int32, (None))

    trainer = Trainer(vocab_size=vocab_len, embedding_size=embedding_size, ini_weight=init_embedding,
                      dropout_rnn_in=dropout_rnn_in, dropout_rnn_out=dropout_rnn_out, hidden_dim=hidden_size,
                      bidirection=True)

    label_onehot = tf.one_hot(y, 4)

    probs,neg_likelihood = trainer.forward(article, msk_a, lst_a, qst, msk_qst, lst_qst, opt, msk_opt, lst_opt,att_msk,label_onehot, if_test)
    one_best = tf.argmax(probs, axis=1)


    neg_likelihood_mean = tf.reduce_mean(neg_likelihood)
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=label_onehot))
    loss = neg_likelihood_mean + class_loss
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=probs,labels=label_onehot)
    #loss = tf.reduce_mean(tf.negative(tf.log(tf.reduce_sum(probs * label_onehot, axis=1))))

    # debug informaiton
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    #print all_test[0][0].shape, all_test[0][1].shape, all_test[0][2]
    #print all_test[0][9]
    debug_output,nl_loss = sess.run([probs,neg_likelihood], {article: all_dev[0][0], msk_a: all_dev[0][1], lst_a: all_dev[0][2],
                                                 qst: all_dev[0][3], msk_qst: all_dev[0][4], lst_qst: all_dev[0][5],
                                                 opt: all_dev[0][6], msk_opt: all_dev[0][7], lst_opt: all_dev[0][8],
                                                 att_msk: all_dev[0][9], y:all_dev[0][10],
                                                 dropout_rnn_in:args.dropout_in, dropout_rnn_out:args.dropout_out, if_test:False})
    print 'debug output:',debug_output.shape,nl_loss

    print tf.trainable_variables()

    decay_steps = 10000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.0001
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        #train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        saver = tf.train.Saver()
        if args.continue_train:
            saver.restore(sess, args.best_model)
        else:
            # Initialize variables
            sess.run(init)
        best_acc = 0
        nupdate = 0

        for epoch in range(args.epoch_num):
            step_idx = 0
            loss_acc = 0
            nl_acc = 0
            start_time = time.time()
            np.random.shuffle(all_train)
            for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
                     mb_mask3, mb_lst3, mb_att,mb_y) in enumerate(all_train):
                #batch x question_num
                [_,loss_this_batch,nl_this_batch] = sess.run([train_op, class_loss,neg_likelihood_mean],
                                               feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                          qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                          opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                          att_msk:mb_att, y:mb_y, dropout_rnn_in:args.dropout_in,
                                                          dropout_rnn_out:args.dropout_out, if_test:False})
                #print loss_this_batch
                step_idx += 1
                loss_acc += loss_this_batch
                nl_acc += nl_this_batch
                nupdate += 1
                if it % 100 == 0:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    logging.info(
                        '-' * 10 + 'Epoch ' + str(epoch) + '-' * 10 + "Average Loss batch " + repr(it) + ":" + str(
                            round(
                                loss_acc / step_idx, 3)) + 'Elapsed time:' + str(round(elapsed, 2)))
                    logging.info('Negative Likelihhod:' + str(nl_acc / step_idx))
                    start_time = time.time()
                    step_idx = 0
                    loss_acc = 0
                    nl_acc = 0

                if nupdate % 1000 == 0:
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
