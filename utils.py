import numpy as np
import json
import cPickle as pkl
import operator
import os
import re
import logging
from collections import Counter
import nltk
from collections import defaultdict
from nltk import tokenize
from nltk import word_tokenize

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


def convert2index(examples, word_dict, opt_num = 4,
                  sort_by_len=True, verbose=True, concat=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    #print examples[2][:10]
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
            for i in range(opt_num):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * opt_num]
                else:
                    op = examples[2][i + idx * opt_num]
                    qsop = examples[3][i + idx * opt_num]
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
                #print inf
                #assert inf in ["middle", "high"]
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

            #assert len(obj["options"][i]) == 4
            for j in range(len(obj["options"][i])):
                #print obj['options'][i][j]
                #questions += [q]
                #documents += [obj["article"]]
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

def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    word_count['a'] = 100000
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}

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

    if test_read_data:
        passages = read_data('RACE/train/middle/')
        print passages[0].keys()

    if test_export_data:
        vocab = load_vocab('RACE/train/vocab_high.pkl')
        passages = read_data('RACE/test/high/')

    if test_load_data:
        vocab = load_vocab('RACE/train/vocab.pkl')
        processed_data = load_data('RACE/train/processed_data.pkl')
        print vocab['the']
        print vocab['saved']
        print processed_data[0]
        print processed_data[1]