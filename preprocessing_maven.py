import json
import pickle

import config_maven as config

tokenizer = config.tokenizer


def build_bert_example(elem, flag):
    words, events = elem[0], elem[1]

    words = ['[CLS]'] + words + ['[SEP]']
    events = ['O'] + events + ['O']

    subword_ids = list()
    spans = list()
    sen_label = [0] * len(config.event_type)
    tok_label = list()

    event_idxes = {}

    for word, event in zip(words, events):
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        spans.append([s, e])

        event_type = event.split(':')[0][2:] if event != 'O' else 'O'
        # if event.split(':')[0][0] == 'I':
        #     event_type = 'O'
        event_type = event_type.replace('&', '_').replace('-', '_')

        if flag == 'all':
            pass
        elif flag == 'salient':
            if event_type in config.unsalient_type:   # only salient type
                event_type = 'O'
        elif flag == 'unsalient':
            if event_type in config.salient_type:   # only salient type
                event_type = 'O'

        t = config.tag2idx[event_type]
        tok_label.append(t)

        if event_type != 'O' and event_type in config.tag2idx:
            i = config.tag2idx[event_type]
            sen_label[i] = 1

            event_idxes.setdefault(i, list())
            for j in range(s, e+1):
                event_idxes[i].append(j)

    return subword_ids, spans, sen_label, event_idxes, words, events, tok_label



def build_examples(data, flag='all'):
    res = []
    for i, elem in enumerate(data):
        subword_ids, spans, label, event_idxes, words, events, tok_labels = build_bert_example(elem, flag)
        res.append([subword_ids, spans, label, event_idxes, i, words, events, tok_labels])
    return res


def read_maven(filename):
    result = []
    with open(filename) as filein:
        for line in filein:
            res = json.loads(line)
            sentences = [x['tokens'] for x in res['content']]
            labels = [['O'] * len(x) for x in sentences]

            events = res['events']
            for event in events:
                ty = event['type']
                for m in event['mention']:
                    sen_id = m['sent_id']
                    s, e = m['offset']
                    for i in range(s, e):
                        if labels[sen_id][i] != 'O':
                            print('Here')
                        labels[sen_id][i] = 'B-' + ty
                    for i in range(s+1, e):
                        labels[sen_id][i] = 'I-' + ty
            for sentence, label in zip(sentences, labels):
                result.append([sentence, label])
    return result



if __name__ == '__main__':

    train = read_maven('data/MAVEN/train.jsonl')
    dev = read_maven('data/MAVEN/valid.jsonl')

    print(len(train), len(dev))

    event_no = 0
    token_no = 0
    event_set = []
    for elem in dev:
        # print(elem[0])
        token_no += len(elem[0])
        for label in elem[1]:
            if label != 'O' and label[0] == 'B':
                event_no += 1
    print(token_no)
    print(event_no)
    
    # print(list(set(event_set)))
    # print(train[0])

    data = {}
    data['train'] = build_examples(train)
    data['val'] = build_examples(dev)
    data['test'] = data['val']

    print(data['train'][0])

    f = open('data/data_maven.pk','wb')
    pickle.dump(data, f)
    f.close()


    data = {}
    data['train'] = build_examples(train, 'salient')
    data['val'] = build_examples(dev, 'salient')
    data['test'] = data['val']

    print(data['train'][0])

    f = open('data/data_maven_salient.pk','wb')
    pickle.dump(data, f)
    f.close()


    data = {}
    data['train'] = build_examples(train, 'unsalient')
    data['val'] = build_examples(dev, 'unsalient')
    data['test'] = data['val']

    print(data['train'][0])

    f = open('data/data_maven_unsalient.pk','wb')
    pickle.dump(data, f)
    f.close()
