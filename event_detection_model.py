import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss

from torch.nn import CrossEntropyLoss
from allennlp.modules.span_extractors import EndpointSpanExtractor

from transformers import BertModel


def get_type_rep():
    import config
    from allennlp.common.util import pad_sequence_to_length
    
    bert = BertModel.from_pretrained(config.bert_dir)

    text = [['[CLS]'] + x.split('_')[1:] for x in config.event_type]
    sub_idx = [config.tokenizer.convert_tokens_to_ids(x) for x in text]
    sub_idx = list(map(lambda x: pad_sequence_to_length(x, 5), sub_idx))
    sub_idx = torch.LongTensor(sub_idx)

    _, rep = bert(sub_idx)
    return rep


class SentenceClassification(nn.Module):
    def __init__(self, bert_dir, y_num, rep):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir)
        # self.type_rep = type_rep
        # self.type_rep.requires_grad = False
        self.fc = nn.Linear(self.bert.config.hidden_size, self.y_num)

    def getName(self):
        return self.__class__.__name__

    def compute_logits(self, data_x, bert_mask):
        outputs = self.bert(data_x, attention_mask=bert_mask)
        bert_enc = outputs[1]

        logits = self.fc(bert_enc)

        # dim1, dim3 = bert_enc.size()
        # dim2 = self.y_num
        # bert_enc = bert_enc.unsqueeze(1).expand_as(torch.randn(dim1, dim2, dim3))
        # type_rep_x = self.type_rep.unsqueeze(0).expand_as(torch.randn(dim1, dim2, dim3))
        # bert_enc = bert_enc + type_rep_x
        # logits = self.fc(bert_enc)
        # logits = logits.squeeze(-1)

        return logits


    def forward(self, data_x, bert_mask, data_y):
        logits = self.compute_logits(data_x, bert_mask)
        loss_fct = MultiLabelSoftMarginLoss()
        loss = loss_fct(logits, data_y)
        return loss
    


class BertED(nn.Module):
    def __init__(self, bert_dir, y_num=None, top_rnns=False):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
       
        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size//2, batch_first=True)

        self.last_n_layer = 1
        self.span_extractor = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size * self.last_n_layer, combination='x+y')
        # self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size * self.last_n_layer)

        self.fc = nn.Linear(self.bert.config.hidden_size * self.last_n_layer, y_num)
        
        
        self.dropout = nn.Dropout(0.5)
    
    def getName(self):
        return self.__class__.__name__

    def compute_logits(self, data_x, bert_mask, data_span):

        outputs = self.bert(data_x, attention_mask=bert_mask)
        
        bert_enc = outputs[0]
        hidden_states = outputs[2]
        
        # bert_enc = hidden_states[2]
        bert_enc = torch.cat(hidden_states[-self.last_n_layer:], dim=-1)

        logits = self.span_extractor(bert_enc, data_span)

        if self.top_rnns:
            logits, _ = self.rnn(logits)
        
        logits = self.fc(logits)

        return logits

    def forward(self, data_x, bert_mask, data_span, data_y):
        # print(data_span)
        
        logits = self.compute_logits(data_x, bert_mask, data_span)

        ## Normal classification
        loss_fct = CrossEntropyLoss(ignore_index=-1) # ingore -1 
        loss = loss_fct(logits.view(-1, self.y_num), data_y.view(-1))

        return loss
    

    def predict(self, data_x, bert_mask, data_span, sequence_mask):
        
        logits = self.compute_logits(data_x, bert_mask, data_span)
        
        classifications = torch.argmax(logits, -1)
        classifications = list(classifications.cpu().numpy())
        predicts = []
        for classification, mask in zip(classifications, sequence_mask):
            predicts.append(classification[:])

        return predicts


if __name__ == '__main__':
    rep = get_type_rep().detach()


