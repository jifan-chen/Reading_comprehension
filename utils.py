import numpy as np
import json
import cPickle as pkl
import operator
import os
import re
import logging
import nltk
from collections import defaultdict
from nltk import tokenize
from nltk import word_tokenize

#class Mask(object):

#    def __init__(self,num_samples,maxlen):
#        self.mask = np.zeros((num_samples,maxlen))
#        self.last_position = []


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post'):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.zeros((num_samples,maxlen),dtype=int)
    mask = np.zeros((num_samples, maxlen))
    last_position = []

    #print sequences
    #print sequences
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
            mask[idx,:len(trunc)] = np.ones(len(trunc))
            last_position.append(len(trunc) - 1)
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
            mask[idx,-len(trunc):] = np.ones(len(trunc))
            last_position.append(len(trunc) - 1)
        else:
            raise ValueError('Padding type "%s" not understood' % padding)


    assert len(last_position) == len(x)
    #mask = mask.reshape(num_samples,maxlen,1)
    return x,mask,last_position

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(np.float)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask, lengths


def read_data(dir_url):
    passages = []
    for file_name in os.listdir(dir_url):
        #print file_name
        f = open(os.path.join(dir_url,file_name))
        passage = json.load(f)
        passages.append([passage,file_name])
    return passages

def build_vocabulary(passages,maxlen=None):
    #word_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    vocab = defaultdict(int)
    count = 0
    for passage in passages:
        count += 1
        if count % 1000 == 0:
            print count
        article = passage['article']
        questions = passage['questions']
        options = passage['options']
        answers = passage['answers']
        article_sentences = tokenize.sent_tokenize(article)

        for sentence in article_sentences:
            seq = word_tokenize(sentence)
            for word in seq:
                vocab[word.lower()] += 1

        for q in questions:
            seq = word_tokenize(q)
            for word in seq:
                vocab[word.lower()] += 1

        for ops in options:
            for one_option in ops:
                seq = word_tokenize(one_option)
                for word in seq:
                    vocab[word.lower()] += 1

    sorted_vocab = sorted(vocab.items(),key=operator.itemgetter(1),reverse=True)

    #print sorted_vocab[:100]
    for i in range(len(sorted_vocab)):
        sorted_vocab[i] = list(sorted_vocab[i])
        sorted_vocab[i][1] = i+1
    #print sorted_vocab[:100]

    if maxlen:
        vocab = dict(sorted_vocab[:maxlen])
    else:
        vocab = dict(sorted_vocab)

    return vocab
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

    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[4])):
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
        #print i
        new_in_x3 += i
        new_in_x4 += j
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, new_in_x4, in_y

def load_data(in_file, max_example=None, relabeling=True):

    documents = []
    questions = []
    answers = []
    options = []
    qs_op = []
    question_belong = []
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
            #documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            for j in range(4):
                #print obj['options'][i][j]
                #questions += [q]
                documents += [obj["article"]]
                qs_op += [q + " " + obj['options'][i][j]]
            options += obj["options"][i]
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
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options,qs_op,answers)

def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def gen_examples(x1, x2, x3, x4, y, batch_size, concat=False):
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
        #mb_x2, mb_mask2, mb_lst2 = utils.pad_sequences(mb_x2)
        #mb_x3, mb_mask3, mb_lst3 = utils.pad_sequences(mb_x3)
        mb_x4, mb_mask4, mb_lst4 = pad_sequences(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_lst1, mb_x4, mb_mask4, mb_lst4, mb_y))
    return all_ex

def export_processed_data(passages,vocab,out_url):
    count = 0
    processed_data = []
    pattern = '[_]'
    for passage,file_name in passages:
        count += 1
        print 'Number of passage processed:' + str(count)
        article = passage['article']
        questions = passage['questions']
        options = passage['options']
        answers = passage['answers']
        article_sentences = tokenize.sent_tokenize(article)

        converted_article = []
        #print article_sentences
        for sentence in article_sentences:
            #seq = re.sub(pattern,'',sentence)
            seq = word_tokenize(sentence)
            for i,word in enumerate(seq):
                try:
                    seq[i] = vocab[word.lower()]
                except KeyError:
                    seq[i] = 0
            converted_article.append(seq)
        #converted_article.sort(key=len)

        converted_questions = []
        for q in questions:
            #q = re.sub(pattern, '', q)
            seq = word_tokenize(q)
            for i, word in enumerate(seq):
                try:
                    seq[i] = vocab[word.lower()]
                except KeyError:
                    seq[i] = 0
            converted_questions.append(seq)
        #converted_questions.sort(key=len)

        converted_options = []
        #print options
        for ops in options:
            tmp = []
            for one_option in ops:
                #one_option = re.sub(pattern, '', one_option)
                seq = word_tokenize(one_option)
                for i, word in enumerate(seq):
                    try:
                        seq[i] = vocab[word.lower()]
                    except KeyError:
                        seq[i] = 0
                tmp.append(seq)
            converted_options.append(tmp)


        for i,ans in enumerate(answers):
            if ans == 'A':
                answers[i] = 0
            elif ans == 'B':
                answers[i] = 1
            elif ans == 'C':
                answers[i] = 2
            else:
                answers[i] = 3

        padded_article, mask_a, lastpst_a = pad_sequences(converted_article, padding='post', truncating='post')
        padded_question = []
        padded_option = []

        '''
        for i in range(len(converted_questions)):
            #print converted_questions
            #print converted_options
            q, mask_q, lastpst_q = pad_sequences([converted_questions[i]], padding='post', truncating='post')
            o, mask_o, lastpst_o = pad_sequences(converted_options[i], padding='post', truncating='post')
            padded_question.append([q,mask_q,lastpst_q])
            padded_option.append([o,mask_o,lastpst_o])
        '''
        try:
            q,mask_q,lastpst_q = pad_sequences(converted_questions,padding='post',truncating='post')
            all_options = []
            for opt in converted_options:
                all_options += opt
            #all_options.sort(key=len)
            o,mask_o,lastpst_o = pad_sequences(all_options,padding='post',truncating='post')
        except ValueError:
            print q,o
            continue

        #print len(padded_question)
        converted_data = {'article': [padded_article,mask_a,lastpst_a],
                          'questions': [q,mask_q,lastpst_q],
                          'answers': answers,
                          'options': [o,mask_o,lastpst_o],
                          'article_id': file_name}
        #rint converted_data['article']

        processed_data.append(converted_data)

    pkl.dump(processed_data,open(out_url,'w'))

def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = np.random.uniform(-0.1,0.1,[num_words,dim])
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
                mu = embeddings[word_dict[sp[0]]].mean()
                #print embeddings[word_dict[sp[0]]]
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu += mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings

def export_vocab(vocab,out_url):
    pkl.dump(vocab, open(out_url, 'w'))

def load_vocab(vocab_url):
    return pkl.load(open(vocab_url))

if __name__ == '__main__':
    test_padding_sentence = False
    test_read_data = False
    test_load_data = False
    test_export_data = True

    if test_padding_sentence:
        t= [[1,2,3],[1],[1,2,3,4],[3,4,5,6,7]]
        padded_sentences,mask,last_position = pad_sequences(t,padding='post',truncating='post')
        print padded_sentences
        print mask.shape
        print last_position

    if test_read_data:
        passages = read_data('RACE/train/middle/')
        print passages[0].keys()

        test_build_vocab = True

        if test_build_vocab:
            maxlen = 50000
            vocab = build_vocabulary(passages,50000)
            export_vocab(vocab,'RACE/train/vocab_high.pkl')

    if test_export_data:
        vocab = load_vocab('RACE/train/vocab_high.pkl')
        passages = read_data('RACE/test/high/')
        export_processed_data(passages, vocab, 'RACE/test/processed_data_high.pkl')

    if test_load_data:
        vocab = load_vocab('RACE/train/vocab.pkl')
        processed_data = load_data('RACE/train/processed_data.pkl')
        print vocab['the']
        print vocab['saved']
        print processed_data[0]
        print processed_data[1]