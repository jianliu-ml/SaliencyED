import pickle
import numpy

import config_maven as config
# import config

if __name__ == '__main__':
    filename = 'maven_salience_ind.pk'
    # filename = 'data_salience_ind.pk'

    data = pickle.load(open(filename, 'rb'))
    
    sta = {}

    for elem in data:
        res, data_x, appendix = elem
        for e, d, p in zip(res, data_x, appendix):
            x = p[0] # {30: [7, 8], 7: [15]}
            if x:
                # print(p)
                words = config.tokenizer.convert_ids_to_tokens(d)
                words = list(filter(lambda x: x!='[PAD]', words))

                probs = res[e]['grad_input_1']
                probs = numpy.around(probs, 3)
                probs = [ x / sum(probs) for x in probs]

                for z in x:
                    p = sum([probs[i] if i < len(probs) else 0.0 for i in x[z]])
                    sta.setdefault(z, [])
                    sta[z].append(p)

                for idx, (w, p) in enumerate(zip(words, probs)):
                    t = ' '
                    for l in x:
                        if idx in x[l]:
                            t = config.idx2tag[l]   # for ACE should add 1
                    temp = [idx, w, ' ', ' ', t, p]
                    temp = [str(i) for i in temp]
                #     print('\t'.join(temp))
                # print('---')

    x = []
    for ev_id in sta:
        x.append([config.idx2tag[ev_id], sum(sta[ev_id]) / len(sta[ev_id])])    # for ACE should add 1
    x = sorted(x, key=lambda x: x[1])
    for i, e in x:
        print(i, e)