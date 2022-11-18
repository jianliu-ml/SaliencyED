import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import warnings
warnings.filterwarnings('ignore')
import pickle
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config_maven as config

from dataset import Dataset
from model import BertED
from utils import save_model, load_model


def load_dataset():
    filename = 'data/data_maven.pk'
    data = pickle.load(open(filename, 'rb'))
    return data['train'], data['val'], data['test']


def evaluate_ed(model, device, testset):

    with torch.no_grad():
        golds = []
        predicts = []

        for batch in testset.get_tqdm(device, False):
            data_x, data_x_mask, data_labels, data_tok_labels, data_span, appendix = batch
            logits = model.compute_logits(data_x, data_x_mask, data_span)
            logits = torch.argmax(logits, -1)

            golds.extend(data_tok_labels.cpu().numpy())
            predicts.extend(logits.cpu().numpy())
    
        all_golds = []
        all_preds = []

        for gold, pred in zip(golds, predicts):
            for i, (g, p) in enumerate(zip(gold, pred)):
                if g == -1:
                    break
                all_golds.append(g)
                all_preds.append(p)
        
        print('Overall Micro', precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average='micro'))
        print('Overall Macro', precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average='macro'))
        ps, rs, f1s, _ = precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average=None)
        # for i, (p, r, f) in enumerate(zip(ps, rs, f1s)):
        #     print(config.idx2tag[i+1], p, r, f)
        print(sum(f1s) / len(f1s))

        print('Salient micro', precision_recall_fscore_support(all_golds, all_preds, labels=config.salient_type_set, average='micro'))
        print('Salient macro', precision_recall_fscore_support(all_golds, all_preds, labels=config.salient_type_set, average='macro'))
        print('Unsalient micro', precision_recall_fscore_support(all_golds, all_preds, labels=config.unsalient_type_set, average='micro'))
        print('Unsalient macro', precision_recall_fscore_support(all_golds, all_preds, labels=config.unsalient_type_set, average='macro'))


def evaluate_ed_sen(model, device, testset):

    with torch.no_grad():
        golds = []
        predicts = []

        for batch in testset.get_tqdm(device, False):
            data_x, data_x_mask, data_labels, data_tok_labels, data_span, appendix = batch
            logits = model.compute_logits(data_x, data_x_mask, data_span)
            logits = torch.argmax(logits, -1)

            golds.extend(data_tok_labels.cpu().numpy())
            predicts.extend(logits.cpu().numpy())
    
    
        n_gold, n_predict, n_correct = 0, 0, 0
        for gold, pred in zip(golds, predicts):

            temp_p = [0] * (len(config.idx2tag) + 1)
            temp_g = [0] * (len(config.idx2tag) + 1)
            temp_p, temp_g = np.asarray(temp_p), np.asarray(temp_g)

            for i, (g, p) in enumerate(zip(gold, pred)):

                if g == -1:
                    break
                if g > 0:
                    temp_g[g] = 1
                if p > 0:
                    temp_p[p] = 1

            n_gold += sum(temp_g)
            n_predict += sum(temp_p)
            n_correct += sum(temp_g * temp_p)
        
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
        print(n_gold, n_predict, n_correct, p, r, f1)



def evaluate_two(model1, model2, device, testset):

    with torch.no_grad():
        golds = []
        predicts1 = []
        predicts2 = []

        predicts1_pro = []
        predicts2_pro = []

        for batch in testset.get_tqdm(device, False):
            data_x, data_x_mask, data_labels, data_tok_labels, data_span, appendix = batch

            golds.extend(data_tok_labels.cpu().numpy())

            logits1 = model1.compute_logits(data_x, data_x_mask, data_span)
            logits1 = torch.softmax(logits1, -1)
            index1 = torch.argmax(logits1, -1)
            predicts1.extend(index1.cpu().numpy())

            temp = torch.gather(logits1, -1, index1.unsqueeze(-1)).squeeze(-1)
            predicts1_pro.extend(temp.cpu().numpy())

            logits2 = model2.compute_logits(data_x, data_x_mask, data_span)
            logits2 = torch.softmax(logits2, -1)
            index2 = torch.argmax(logits2, -1)  
            predicts2.extend(index2.cpu().numpy())

            temp = torch.gather(logits2, -1, index2.unsqueeze(-1)).squeeze(-1)
            predicts2_pro.extend(temp.cpu().numpy())



    
        all_golds = []
        all_pred1s = []
        all_pred2s = []

        for gold, pred1, prd1_pro, pred2, prd2_pro in zip(golds, predicts1, predicts1_pro, predicts2, predicts2_pro):
            for i, (g, p1, p1_p, p2, p2_p) in enumerate(zip(gold, pred1, prd1_pro, pred2, prd2_pro)):
                if g == -1:
                    break

                all_golds.append(g)
                all_pred1s.append((p1, p1_p))
                all_pred2s.append((p2, p2_p))

        def _c_label(p1, p2):
            if p1[0] == 0:
                return p2[0]
            elif p2[0] == 0:
                return p1[0]
            
            if p1[1] > p2[1]:
                return p1[0]
            return p2[0]

        all_preds = [_c_label(p1, p2) for p1, p2 in zip(all_pred1s, all_pred2s)]

        print('Overall Micro', precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average='micro'))
        print('Overall Macro', precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average='macro'))
        ps, rs, f1s, _ = precision_recall_fscore_support(all_golds, all_preds, labels=list(range(1, len(config.idx2tag))), average=None)
        for i, (p, r, f) in enumerate(zip(ps, rs, f1s)):
            print(config.idx2tag[i+1], p, r, f)
        print(sum(f1s) / len(f1s))

        # print('Salient Micro', precision_recall_fscore_support(all_golds, all_preds, labels=config.salient_type_set, average='micro'))
        # print('Unsalient Micro', precision_recall_fscore_support(all_golds, all_preds, labels=config.unsalient_type_set, average='micro'))
        print('______')


if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 10
    n_epochs = 10

    train_data, val_data, test_data = load_dataset()
    train_data = list(filter(lambda x: len(x[0])<150, train_data))
    val_data = list(filter(lambda x: len(x[0])<120, val_data))
    

    print(test_data[0])

    train_dataset = Dataset(batch_size, 150, train_data)
    val_dataset = Dataset(batch_size, 150, val_data)
    test_dataset = val_dataset


    model1 = BertED(config.bert_dir, len(config.idx2tag))
    load_model(model1, 'model/maven_salient.ckp')
    model1.to(device)

    model2 = BertED(config.bert_dir, len(config.idx2tag))
    load_model(model2, 'model/maven_unsalient.ckp')
    model2.to(device)

    model1.eval()
    model2.eval()
    evaluate_two(model1, model2, device, test_dataset)

    print('______')
    
    
    # model3 = BertED(config.bert_dir, len(config.idx2tag))
    # load_model(model3, 'model/maven_salient_1.ckp')
    # model3.to(device)

    # model3.eval()
    # evaluate_ed(model3, device, val_dataset)
    # evaluate_ed_sen(model, device, val_dataset)
    
