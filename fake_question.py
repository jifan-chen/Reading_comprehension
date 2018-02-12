import logging
import numpy as np
import json
import os
def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()

        question = question.replace('@placeholder','_')
        options = []
        new_answer = 'A'
        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]

            question = ' '.join(q_words)
            document = ' '.join(d_words)

            entities = entity_dict.values()
            options += [answer]

            if len(entities) < 4:
                f.readline()
                continue

            while True:
                idx = np.random.randint(0,len(entities))
                candidate = entities[idx]
                if not (candidate == answer):
                    options += [candidate]
                if len(options) == 4:
                    break
            np.random.shuffle(options)
            new_answer = unichr(options.index(answer) + ord('A'))

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        num_examples += 1

        json_file = {'article': document,
                     'questions': [question],
                     'options': [options],
                     'answers': [new_answer]}
        #print num_examples
        json.dump(json_file, open(os.path.join('cnn/dev/', str(num_examples) + '.txt'), "w"), indent=4)

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, answers)

if __name__ == '__main__':
    d,q,a = load_data('cnn/dev.txt',1000)
    #for d_,q_,a_ in zip(d,q,a):
        #print 'document:',d_
        #print 'question:',q_
        #print 'answer:',a_