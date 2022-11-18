import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pickle
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config

from dataset import Dataset
from model import SentenceClassification, get_type_rep
from utils import save_model, load_model


def load_dataset():
    filename = 'data/data_ace_all_mask.pk'
    data = pickle.load(open(filename, 'rb'))
    return data['train'], data['val'], data['test']


def evaluate(model, device, testset):
    with torch.no_grad():
        golds = []
        predicts = []

        for batch in testset.get_tqdm(device, False):
            data_x, data_x_mask, data_labels, _, _, appendix = batch
            logits = model.compute_logits(data_x, data_x_mask)
            logits = torch.nn.Sigmoid()(logits)
            golds.extend(data_labels.cpu().numpy())
            predicts.extend(logits.cpu().numpy())
    
    for i in range(10):
        th = i / 10
        n_gold, n_predict, n_correct = 0, 0, 0
        for g, p in zip(golds, predicts):
            n_gold += sum(g)
            n_predict += sum(p >= th)
            n_correct += sum(g * (p > th))
        
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
        print(n_gold, n_predict, n_correct, p, r, f1)

    print('___')
    for i in range(1, len(config.idx2tag)):
        th = 0.3
        n_gold, n_predict, n_correct = 0, 0, 0
        for g, p in zip(golds, predicts):
            g = g[i:i+1]
            p = p[i:i+1]
            n_gold += sum(g)
            n_predict += sum(p >= th)
            n_correct += sum(g * (p > th))
        
        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
        print(config.idx2tag[i], n_gold, n_predict, n_correct, p, r, f1)

    
    n_gold, n_predict, n_correct = 0, 0, 0
    for i in range(1, len(config.idx2tag)):
        if i in config.salient_type_set:
            continue
        th = 0.3
        for g, p in zip(golds, predicts):
            g = g[i:i+1]
            p = p[i:i+1]
            n_gold += sum(g)
            n_predict += sum(p >= th)
            n_correct += sum(g * (p > th))
        
    p = n_correct / n_predict
    r = n_correct / n_gold
    f1 = 2 * p * r / (p + r)
    print(config.idx2tag[i], n_gold, n_predict, n_correct, p, r, f1)
        

if __name__ == '__main__':

    device = 'cuda'

    batch_size = 10

    train_data, val_data, test_data = load_dataset()

    train_dataset = Dataset(batch_size, 150, train_data)
    val_dataset = Dataset(batch_size, 100, val_data)
    test_dataset = Dataset(batch_size, 80, test_data)

    rep = get_type_rep().detach_().to(device)
    model = SentenceClassification(config.bert_dir, len(config.idx2tag), rep)
    load_model(model, 'model/ace_sentence9.ckp')
    model.to(device)

    model.eval()
    evaluate(model, device, test_dataset)
    
