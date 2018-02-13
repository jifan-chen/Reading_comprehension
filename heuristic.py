import utils
from rouge import Rouge
from utils import *
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords


def remove_stop_words(s):
    return ' '.join([word for word in s.split() if word not in stopwords.words('english')])

def load_data(in_file, max_example=None, relabeling=True):

    documents = []
    questions = []
    answers = []
    options = []
    qs_op = []
    evidences = []
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
            documents += [obj["article"]]
            questions += [q]
            assert len(obj["options"][i]) == 4
            for j in range(4):
                #print obj['options'][i][j]
                qs_op += [q + " " + obj['options'][i][j]]
            options += obj["options"][i]
            evidences += [obj['evidence'][i]]
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
    return (documents, questions, options,qs_op,answers,evidences)


data = utils.load_data('cnn/test/')
rouge = Rouge()
passages = data[0]
questions = data[1]
options =  data[2]
question_options = data[3]
answers = data[4]
#gold_evidences = data[5]
#print gold_evidences
print len(data[0])
print len(data[1])
print len(data[2])
print len(data[3])
print len(data[4])
predicts = []
find_count = 0
gold_count = 0
predict_count = 0
for i in range(len(passages)):
    p = passages[i]
    tkps = tokenize.sent_tokenize(p)
    q = questions[i]
    opts = options[i*4:(i+1)*4]
    qst_opts = question_options[i*4:(i+1)*4]
    evidences = []
    best_scores = []
    if '_' in q.split():
        opts = qst_opts
    #for qstopt in qst_opts:
    for opt in opts:
        best = 0
        #print opt
        for s in tkps:
            #score = rouge.get_scores([qstopt],[s])
            #opt = remove_stop_words(opt)
            #s = remove_stop_words(s)
            score = rouge.get_scores([opt],[s])
            #rouge1 = score[0]['rouge-1']['f']
            rouge1 = score[0]['rouge-1']['f']
            if rouge1 >= best:
                e = s
                best = rouge1
        best_scores.append(best)
        #if e not in evidences:
        #    evidences.append(e)
    print opts
    #print 'evidence:',evidences
    #print 'gold_evidence:',gold_evidences[i]
    print best_scores
    #evidences = ' '.join(evidences)

    '''
    for evd in evidences:
        for gevd in tokenize.sent_tokenize(gold_evidences[i]):
            score_ = rouge.get_scores([evd],[gevd])
            rouge1_ = score_[0]['rouge-1']['f']
            if rouge1_ > 0.6:
                find_count += 1
            #print 'evidence:', evd
            #print 'gold_evidence:',gevd
            #print rouge1_
    gold_count += len(tokenize.sent_tokenize(gold_evidences[i]))
    predict_count += len(evidences)
    '''
    #try:
    #    print find_count / float(gold_count)
    #except ZeroDivisionError:
    #    pass
    if 'not' in q.split():
        print q
        ans = best_scores.index(min(best_scores))
    else:
        ans = best_scores.index(max(best_scores))
    predicts.append(ans)

#print 'recall:',find_count / float(gold_count)
#print 'precision:',find_count/ float(predict_count)
print answers
print predicts
print accuracy_score(answers,predicts)
    #score = rouge.get_scores([opts[1]],[p])
    #print score