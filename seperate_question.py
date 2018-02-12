import os
import json
import copy
from nltk import sent_tokenize, word_tokenize
from rouge import Rouge
#from nltk import tokenize

rouge = Rouge()

def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()

def seperate_by_fact():
    difficulty_set = ["middle", "high"]
    # difficulty_set = ["high"]
    data = "none_fact_questions_strict/"
    raw_data = "RACE/data/"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    num_que = 0
    for data_set in ["train", 'dev', 'test']:
        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            new_data_path = os.path.join(data, data_set, d)
            deleted_questions = 0
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                if inf == '.DS_Store':
                    continue
                # print inf,new_raw_data_path
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                new_obj = copy.deepcopy(obj)
                obj["article"] = obj["article"].replace("\\newline", "\n")
                obj["article"] = tokenize(obj["article"])
                tkps = sent_tokenize(obj["article"])
                avg_article_length += obj["article"].count(" ")
                deleted = 0
                for i in range(len(obj["questions"])):
                    num_que += 1
                    obj["questions"][i] = tokenize(obj["questions"][i])
                    avg_question_length += obj["questions"][i].count(" ")
                    delete = False
                    #print ord(obj["answers"][i]) - ord('A')
                    for k in range(4):

                        for s in tkps:
                            score = rouge.get_scores([obj["options"][i][k]], [s])
                            rouge1 = score[0]['rouge-1']['p']
                            # print rouge1
                            if rouge1 >= 0.9 and k == ord(obj["answers"][i]) - ord('A'):
                                delete = True

                    if delete:
                        new_obj['questions'].pop(i - deleted)
                        new_obj['options'].pop(i - deleted)
                        new_obj['answers'].pop(i - deleted)
                        deleted += 1
                deleted_questions += deleted
                if len(new_obj['questions']):
                    json.dump(new_obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
            print deleted_questions

def seperate_by_type():
    keyword_sets = ['what', 'when', 'who', 'why', 'how', 'which', 'where']
    difficulty_set = ["middle", "high"]
    # difficulty_set = ["high"]
    data = "wh_questions/"
    raw_data = "RACE/data/"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    num_que = 0
    for data_set in ["train", 'dev', 'test']:
        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            new_data_path = os.path.join(data, data_set, d)
            deleted_questions = 0
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                if inf == '.DS_Store':
                    continue
                # print inf,new_raw_data_path
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                new_obj = copy.deepcopy(obj)
                obj["article"] = obj["article"].replace("\\newline", "\n")
                obj["article"] = tokenize(obj["article"])
                avg_article_length += obj["article"].count(" ")
                deleted = 0
                for i in range(len(obj["questions"])):
                    num_que += 1
                    delete = True
                    obj["questions"][i] = tokenize(obj["questions"][i])
                    avg_question_length += obj["questions"][i].count(" ")
                    for word in obj["questions"][i].split():
                        #print word
                        if word in keyword_sets:
                            #print word
                            delete = False
                    if delete:
                        new_obj['questions'].pop(i - deleted)
                        new_obj['options'].pop(i - deleted)
                        new_obj['answers'].pop(i - deleted)
                        deleted += 1
                deleted_questions += deleted
                if len(new_obj['questions']):
                    json.dump(new_obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
            print deleted_questions

if __name__ == "__main__":
    seperate_by_fact()
    #seperate_by_type()