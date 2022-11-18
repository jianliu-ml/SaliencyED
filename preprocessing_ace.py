import json
import pickle

import config

tokenizer = config.tokenizer


def build_bert_example(elem, flag):
    words, events = elem[1], elem[4]

    words = ['[CLS]'] + words + ['[SEP]']
    events = ['O'] + events + ['O']

    subword_ids = list()
    spans = list()
    sen_label = [0] * len(config.event_type)
    tok_label = list()

    event_idxes = {}

    for word, event in zip(words, events):


        event_type = event.split(':')[0][2:] if event != 'O' else 'O'   #### how to process multi-word trigger ?
        if event.split(':')[0][0] == 'I':
            event_type = 'O'
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
            
            # word = '[MASK]'

        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        spans.append([s, e])

    return subword_ids, spans, sen_label, event_idxes, words, events, tok_label



def build_examples(data, flag='all'):
    res = []
    for i, elem in enumerate(data):
        subword_ids, spans, label, event_idxes, words, events, tok_labels = build_bert_example(elem, flag)
        res.append([subword_ids, spans, label, event_idxes, i, words, events, tok_labels])
    return res




if __name__ == '__main__':

    filename = 'data/data.pickle'
    data = pickle.load(open(filename, 'rb'))

    data['train'] = build_examples(data['train'], 'all')
    data['val'] = build_examples(data['val'], 'all')
    data['test'] = build_examples(data['test'], 'all')

    f = open('data/data_ace_all_mask.pk','wb')
    pickle.dump(data, f)
    f.close()
    

    filename = 'data/data.pickle'
    data = pickle.load(open(filename, 'rb'))

    data['train'] = build_examples(data['train'], 'all')
    data['val'] = build_examples(data['val'], 'all')
    data['test'] = build_examples(data['test'], 'all')

    f = open('data/data_ace_all.pk','wb')
    pickle.dump(data, f)
    f.close()


    filename = 'data/data.pickle'
    data = pickle.load(open(filename, 'rb'))

    data['train'] = build_examples(data['train'], 'salient')
    data['val'] = build_examples(data['val'], 'salient')
    data['test'] = build_examples(data['test'], 'salient')

    f = open('data/data_ace_salient.pk','wb')
    pickle.dump(data, f)
    f.close()


    filename = 'data/data.pickle'
    data = pickle.load(open(filename, 'rb'))

    data['train'] = build_examples(data['train'], 'unsalient')
    data['val'] = build_examples(data['val'], 'unsalient')
    data['test'] = build_examples(data['test'], 'unsalient')

    f = open('data/data_ace_unsalient.pk','wb')
    pickle.dump(data, f)
    f.close()

