import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pickle
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config_maven as config

from dataset import Dataset
from model import SentenceClassification, get_type_rep
from utils import save_model


def load_dataset():
    filename = 'data/data_maven.pk'
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
    
    n_gold, n_predict, n_correct = 0, 0, 0
    th = 0.5
    for g, p in zip(golds, predicts):
        n_gold += sum(g)
        n_predict += sum(p >= th)
        n_correct += sum(g * (p > th))
    
    p = n_correct / n_predict
    r = n_correct / n_gold
    f1 = 2 * p * r / (p + r)
    print(n_gold, n_predict, n_correct, p, r, f1)



if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 10
    n_epochs = 10

    train_data, val_data, test_data = load_dataset()

    train_dataset = Dataset(batch_size, 150, train_data)
    test_dataset = Dataset(batch_size, 100, test_data)

    rep = get_type_rep().detach_().to(device)

    model = SentenceClassification(config.bert_dir, len(config.idx2tag), rep)
    model.to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)

    for idx in range(n_epochs):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, data_x_mask, data_labels, _, _, appendix = batch
            loss = model(data_x, data_x_mask, data_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.zero_grad()
        
        save_model(model, 'model/maven_sentence%d.ckp' % (idx))

        # evaluate
        # model.eval()
        # evaluate(model, device, test_dataset)

