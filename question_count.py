import tensorflow as tf
import AttentionLayer_tf
from Encoder_tf import RNN_encoder
import utils
from rouge import Rouge
from utils import *
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

rouge = Rouge()

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
        #self.option_dot_product = AttentionLayer_tf.DotProductAttention(self.weight_mtx_dim,'W_dotprod')
        self.p2opt_attention = AttentionLayer_tf.BilinearAttentionP2QA(self.weight_mtx_dim,'W_p2opt')
        self.opt2p_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_opt2p')
        self.q2opt_attention = AttentionLayer_tf.BilinearAttentionO2P(self.weight_mtx_dim,'W_q2opt')
        self.e2opt_attention = AttentionLayer_tf.BilinearAttentionP2Q(self.weight_mtx_dim,'W_e2opt')
        self.e2opt_dot = AttentionLayer_tf.BilinearDotM2M(self.weight_mtx_dim,'W_e2opt_dot')

        self.passage_encoder = RNN_encoder(hidden_dim,'passage_encoder',bidirection,keep_prob=0.8)
        self.question_encoder = RNN_encoder(hidden_dim,'question_encoder',bidirection,keep_prob=0.8)
        self.option_encoder = RNN_encoder(hidden_dim,'option_encoder',bidirection,keep_prob=0.8)
        self.evidence_encoder = RNN_encoder(hidden_dim,'evidence_encoder',bidirection,keep_prob=0.8)

        self.psg_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'passage')
        self.qst_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'question')
        self.opt_selfatt = AttentionLayer_tf.SelfAttention(self.weight_mtx_dim,'option')
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
        opt_ht = tf.reshape(opt_ht, [-1, self.option_number, self.weight_mtx_dim])
        evd_ht = tf.reshape(evd_ht, [-1, self.option_number, self.weight_mtx_dim])

        qst_ht = tf.expand_dims(qst_ht, 1)
        qstopt_ht = opt_ht + qst_ht

        probs = self.e2opt_dot.score(evd_ht,qstopt_ht)
        return probs

def extract_evidence(psg,opts):
    #print type(psg)
    tkps = tokenize.sent_tokenize(psg)
    evidences = []
    best_scores = []
    for opt in opts:
        best = 0
        #print opt
        for s in tkps:
            score = rouge.get_scores([opt],[s])
            #rouge1 = score[0]['rouge-1']['f']
            rouge1 = score[0]['rouge-1']['f']
            if rouge1 >= best:
                e = s
                best = rouge1
        best_scores.append(best)
        evidences.append(e)
    return evidences,best_scores

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
                #print obj['options'][i][j]
                qs_op += [q + " " + obj['options'][i][j]]
            options += obj["options"][i]
            evds,scores = extract_evidence(obj['article'],obj["options"][i])
            evidences += evds
            best_scores += scores

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

def remove_stop_words(s):
    return ' '.join([word for word in s.split() if word not in stopwords.words('english')])

data = load_data('RACE/data/dev/middle')
passages = data[0]
questions = data[1]
options =  data[2]
question_options = data[3]
answers = data[6]

print len(data[0])
print len(data[1])
print len(data[2])
print len(data[3])
print len(data[4])
predicts = []
find_count = 0
gold_count = 0
predict_count = 0

wh_question = 0
fill_question = 0
wh_index = []
fill_index = []
wh_correct = 0
fill_correct = 0
other_correct = 0
correct_indexs_heuristc = []

keyword_sets = ['what','when','who','why','how','which','where']

for i in range(len(passages)):
    p = passages[i]
    tkps = tokenize.sent_tokenize(p)
    q = questions[i]
    q = q.lower()
    opts = options[i*4:(i+1)*4]
    qst_opts = question_options[i*4:(i+1)*4]
    evidences = []
    best_scores = []
    if '_' in q.split():
        opts = qst_opts
    for opt in opts:
        best = 0
        #print opt
        for s in tkps:
            score = rouge.get_scores([opt],[s])
            rouge1 = score[0]['rouge-1']['f']
            if rouge1 >= best:
                e = s
                best = rouge1
        best_scores.append(best)

    #print opts
    #print best_scores
    #    pass
    if 'not' in q.split():
        #print q
        ans = best_scores.index(min(best_scores))
    else:
        ans = best_scores.index(max(best_scores))

    gold = answers[i]
    correct = (gold == ans)


    for w in q.split():
        if w in keyword_sets:
            wh_question += 1
            wh_correct += correct
            wh_index.append(i)
            if correct:
                correct_indexs_heuristc.append(i)
            break
        elif w == '_':
            fill_question += 1
            fill_correct += correct
            fill_index.append(i)
            if correct:
                correct_indexs_heuristc.append(i)
            break

    predicts.append(ans)

print 'wh total:',wh_question
print 'fill_total:',fill_question
print 'wh indexs:',wh_index
print 'fill indexs:',fill_index

print 'Heuristc wh_correct:',wh_correct
print 'Heuristc fill_correct:',fill_correct

#print 'other_correct:',other_correct

print answers
print predicts
print accuracy_score(answers,predicts)


print '-'*20,'Loading NN Moel','-'*20

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

test_data = data
test_x1, test_x2, test_x3, test_x4, test_x5,test_x6, test_y = convert2index(test_data, vocab,sort_by_len=False)
all_test = gen_examples(test_x1, test_x2, test_x3, test_x4, test_x5,test_x6, test_y,32)

saver = tf.train.Saver()
attentions = []

with tf.Session() as sess:
    saver.restore(sess,"model/best_model")
    #new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('model/'))

    predicts = []
    gold = []
    step_idx = 0
    loss_acc = 0

    for it, (mb_x1, mb_mask1, mb_lst1, mb_x2, mb_mask2, mb_lst2, mb_x3,
             mb_mask3, mb_lst3, mb_x4, mb_mask4, mb_lst4, mb_s,mb_y) in enumerate(all_test):
        # Evaluate on the dev set
        train_correct = 0
        [pred_this_instance, loss_this_batch] = \
            sess.run([one_best, loss], feed_dict={article: mb_x1, msk_a: mb_mask1, lst_a: mb_lst1,
                                                  qst: mb_x2, msk_qst: mb_mask2, lst_qst: mb_lst2,
                                                  opt: mb_x3, msk_opt: mb_mask3, lst_opt: mb_lst3,
                                                  evd: mb_x4, msk_evd: mb_mask4, lst_evd: mb_lst4,
                                                  evd_score: mb_s, y: mb_y, dropout_rate: 1.0})
        predicts += list(pred_this_instance)
        gold += mb_y
        step_idx += 1
        loss_acc += loss_this_batch

    dev_acc = accuracy_score(gold, predicts)
    print dev_acc

wh_correct_nn = 0
fill_correct_nn = 0
correct_indexs_nn = []

for i in wh_index:
    wh_correct_nn += predicts[i] == gold[i]
    if predicts[i] == gold[i]:
        correct_indexs_nn.append(i)
for i in fill_index:
    fill_correct_nn += predicts[i] == gold[i]
    if predicts[i] == gold[i]:
        correct_indexs_nn.append(i)

print 'NN wh_correct:',wh_correct_nn
print 'NN fill_correct:',fill_correct_nn
print len(correct_indexs_heuristc)
print len(correct_indexs_nn)
print correct_indexs_heuristc
print correct_indexs_nn

same_count = 0
common_indexs = set()
for h in correct_indexs_heuristc:
    if h in correct_indexs_nn:
        same_count += 1
        common_indexs.add(h)
unique_indexs = set(correct_indexs_heuristc) - common_indexs
print unique_indexs
print common_indexs
print 'Both correct:',same_count