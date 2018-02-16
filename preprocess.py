import logging
import os
import json
import re
from nltk import sent_tokenize, word_tokenize
from nltk.tag import StanfordNERTagger
from collections import Counter
from nltk.tokenize.stanford import StanfordTokenizer

#word_tokenizer = StanfordTokenizer('RACE/stanford-postagger-3.8.0.jar')
tagger = StanfordNERTagger('RACE/english.all.3class.distsim.crf.ser.gz','RACE/stanford-ner.jar')

def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()

def build_entity_dict(passage):

    p = []
    for sent in sent_tokenize(passage):
        p += word_tokenize(sent)
    #print p
    tagged_sentence = tagger.tag(p)
    #print tagged_sentence
    entity_dict = {}
    entity_count = Counter()
    entity = ''
    for i in range(len(tagged_sentence)-1):
        current_word = tagged_sentence[i][0]
        current_tag = tagged_sentence[i][1]
        next_tag = tagged_sentence[i+1][1]
        if current_tag in ['LOCATION', 'PERSON', 'ORGANIZATION']:
            entity += ' '+ current_word
            if not (next_tag == current_tag):
                if entity.strip() not in entity_dict.keys():
                    entity_count[current_tag] += 1
                    entity_dict[entity.strip()] = current_tag + str(entity_count[current_tag])
                    #print entity
                    entity = ''
                else:
                    entity = ''
    print entity_dict
    #print entity_count
    return entity_dict,' '.join(p)


def entity_anonymous(sentence,entity_dict=None,passage=False):
    for key,value in entity_dict.items():
        #sentence = re.sub(key,value,sentence)
        sentence = sentence.replace(key,value)
    if passage:
        return sentence.lower()
    else: return ' '.join(word_tokenize(sentence)).lower()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    difficulty_set = ["middle", "high"]
    difficulty_set = ['middle']
    #difficulty_set = ['high']
    data = "processed_data/"
    raw_data = "RACE_raw/"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    num_que = 0
    for data_set in ["dev", "test"]:
        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            new_data_path = os.path.join(data, data_set, d)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                cnt += 1
                if cnt % 100 == 0:
                    logging.info(str(cnt)+' processed')
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                obj["article"] = obj["article"].replace("\\newline", "\n")
                entity_dict,p = build_entity_dict(passage = obj["article"])
                obj["article"] = entity_anonymous(p,entity_dict,passage=True)
                avg_article_length += obj["article"].count(" ")
                for i in range(len(obj["questions"])):
                    num_que += 1
                    obj["questions"][i] = entity_anonymous(obj["questions"][i],entity_dict)
                    avg_question_length += obj["questions"][i].count(" ")
                    for k in range(4):
                        obj["options"][i][k] = entity_anonymous(obj["options"][i][k],entity_dict)
                        avg_option_length += obj["options"][i][k].count(" ")
                json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
    '''print "avg article length", avg_article_length * 1. / cnt
    print "avg question length", avg_question_length * 1. / num_que
    print "avg option length", avg_option_length * 1. / (num_que * 4)'''
