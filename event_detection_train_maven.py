import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import torch

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config_maven as config

bert_dir = config.bert_dir
num_class = len(config.idx2tag)

from dataset import Dataset
from model import BertED
from utils import save_model
from ed_predict import evaluate_ed

# data_file_name = 'data/data_maven_salient.pk'
# model_save_name = 'model/maven_salient'

data_file_name = 'data/data_maven_unsalient.pk'
model_save_name = 'model/maven_unsalient'


def load_dataset():
    filename = data_file_name
    data = pickle.load(open(filename, 'rb'))
    return data['train'], data['val'], data['test']


if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 8
    n_epochs = 10

    train_data, val_data, test_data = load_dataset()
    train_data = list(filter(lambda x: len(x[0])<150, train_data))
    test_data = list(filter(lambda x: len(x[0])<120, test_data))
    print(train_data[0])

    train_dataset = Dataset(batch_size, 150, train_data)
    test_dataset = Dataset(batch_size, 120, test_data)

    model = BertED(bert_dir, num_class)
    model.to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)

    for idx in range(n_epochs):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, data_x_mask, data_labels, data_tok_labels, data_span, appendix = batch
            loss = model(data_x, data_x_mask, data_span, data_tok_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.zero_grad()
        
        save_model(model, '%s_%d.ckp' % (model_save_name, idx))

        # evaluate
        model.eval()
        evaluate_ed(model, device, test_dataset)

