import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertForTokenClassification
from transformers import BertModel
from torch.autograd import Variable
from torchcrf import CRF
import numpy as np


def calculate_confidence(vec, proportions=0.5):
    """
    calculate the value of alpha, the employed metric is GINI index
    :param vec:
    :return:
    """
    a=torch.mul(vec,vec)
    square_sum = torch.sum(a,dim=-1,keepdim=True)
    a=1 - square_sum
    b=a* proportions
    return (1 - square_sum) * proportions

class Token_OTE2TS(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_OTE2TS, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 50
        self.dim_ote_y = 4
        self.dim_ts_y = 7
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'B': ['B-PRO', 'B-CON'],
                           'I': ['I-PRO', 'I-CON'],
                           'E': ['E-PRO', 'E-CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((4, 7))

        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
        ts_tag_vocab = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(1, 2 * self.dim_ote_h))
        self.dropout = nn.Dropout(0.1)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)



    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, BIEO_labels=None, BIEOstance_labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.tokenbert.dropout(sequence_output)  # (B, L, H)
        #sequence_output, _ = self.rnn(sequence_output.view(sequence_output.shape[0], -1, 768))
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        # stm_lm_hs = self.stm_lm(ote_hs)
        # stm_lm_hs = self.dropout(stm_lm_hs)
        #
        # ts_hs,_ = self.gru_ts(ote_hs)
        # stm_lm_ts = self.stm_ts(ts_hs)
        # stm_lm_ts = self.dropout(stm_lm_ts)
        #
        #
        loss_fct = nn.CrossEntropyLoss()
        # p_y_x_ote = self.fc_ote(ote_hs)
        # # loss_ote = loss_fct(
        # #     p_y_x_ote.view(-1, self.dim_ote_y),
        # #     BIEO_labels.view(-1)
        # # )
        # # probability distribution over ts tag
        #
        # p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(ote_hs)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score
        # alpha = calculate_confidence(vec=p_y_x_ote_softmax)
        # # transition score from ote tag to sentiment tag
        # # a = self.transition_scores.unsqueeze(0)
        # # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])
        # ote2ts = torch.matmul(p_y_x_ote, self.transition_scores.to('cuda'))
        # # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        # p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 7]), ote2ts) + torch.mul((1 - alpha),
        #                                                                                                 p_y_x_ts_softmax)
        if labels is not None: # training
            # loss_ote = loss_fct(
            #     p_y_x_ote.view(-1, self.dim_ote_y),
            #     BIEO_labels.view(-1)
            # )
            loss_ts = loss_fct(
                p_y_x_ts.view(-1, self.dim_ts_y),
                BIEOstance_labels.view(-1)
            )
            loss_i = loss_ts
            return loss_i
        else: # inference
            return torch.argmax(p_y_x_ts, dim=-1)

        # #gt = torch.sigmoid(self.W_gate * ts_ht)#权重
        # ts_hs_tilde = []
        # ts_hs_tildes = []
        # for t in range(sequence_output.shape[0]):
        #     for i in range(sequence_output.shape[1]):
        #         if i == 0:
        #             h_tilde_t = ts_hs[t][i].unsqueeze(1)
        #             ts_hs_tilde = h_tilde_t.view(1, -1)
        #         else:
        #             # t-th hidden state for the task targeted sentiment
        #             ts_ht = ts_hs[t][i].unsqueeze(1)
        #             gt = torch.sigmoid(torch.mm(self.W_gate.to('cuda'), ts_ht))
        #             h_tilde_t = gt[0][0] * ts_ht + (1 - gt[0][0]) * h_tilde_tm1
        #             #print(h_tilde_t)
        #         #ts_hs_tilde.append(h_tilde_t.view(1,-1))
        #             ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.view(1,-1)), 0)
        #         h_tilde_tm1 = h_tilde_t
        #     if t==0:
        #         ts_hs_tildes=ts_hs_tilde.unsqueeze(0)
        #     else:
        #         ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)
        #
        # ote_hs = self.dropout(ote_hs)
        # ts_hs_tildes = self.dropout(ts_hs_tildes)
        #
        # loss_fct = nn.CrossEntropyLoss()
        #
        # p_y_x_ote = self.fc_ote(ote_hs)
        # # loss_ote = loss_fct(
        # #     p_y_x_ote.view(-1, self.dim_ote_y),
        # #     BIEO_labels.view(-1)
        # # )
        # # probability distribution over ts tag
        #
        # p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        # p_y_x_ts = self.fc_ts(ts_hs_tildes)
        # p_y_x_ts = F.softmax(p_y_x_ts, dim=-1)  # 13
        # # normalized the score
        # alpha = calculate_confidence(vec=p_y_x_ote_softmax)
        # # transition score from ote tag to sentiment tag
        # #a = self.transition_scores.unsqueeze(0)
        # #b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])
        # ote2ts = torch.matmul(p_y_x_ote, self.transition_scores.to('cuda'))
        # #ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        # a=alpha.expand([sequence_output.shape[0],100,7])
        # b=(1 - alpha).expand([sequence_output.shape[0],100,7])
        # p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0],100,7]), ote2ts) + torch.mul((1 - alpha), p_y_x_ts)
        # # loss_ts = loss_fct(
        # #     p_y_x_ts_tilde.view(-1, self.dim_ts_y),
        # #     BIEOstance_labels.view(-1)
        # # )
        #
        #
        # #loss_i = loss_ote + loss_ts
        #
        # if labels is not None: # training
        #     loss_ote = loss_fct(
        #         p_y_x_ote.view(-1, self.dim_ote_y),
        #         BIEO_labels.view(-1)
        #     )
        #     loss_ts = loss_fct(
        #         p_y_x_ts_tilde.view(-1, self.dim_ts_y),
        #         BIEOstance_labels.view(-1)
        #     )
        #     loss_i = loss_ote + loss_ts
        #     return loss_i
        # else: # inference
        #     return torch.argmax(p_y_x_ote, dim=-1), torch.argmax(p_y_x_ts_tilde, dim=-1)


class Token_OTE2TS2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_OTE2TS2, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 50
        self.dim_ote_y = 4
        self.dim_ts_y = 7
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'B': ['B-PRO', 'B-CON'],
                           'I': ['I-PRO', 'I-CON'],
                           'E': ['E-PRO', 'E-CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((4, 7))

        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
        ts_tag_vocab = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(4, 7))
        self.dropout = nn.Dropout(0.1)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)



    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, BIEO_labels=None, BIEOstance_labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.tokenbert.dropout(sequence_output)  # (B, L, H)
        #sequence_output, _ = self.rnn(sequence_output.view(sequence_output.shape[0], -1, 768))
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)

        ts_hs,_ = self.gru_ts(ote_hs)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)

        alpha = torch.matmul(torch.matmul(stm_lm_hs, self.W_gate), torch.transpose(stm_lm_ts,1,2))

        loss_fct = nn.CrossEntropyLoss()
        p_y_x_ote = self.fc_ote(ote_hs)
        # loss_ote = loss_fct(
        #     p_y_x_ote.view(-1, self.dim_ote_y),
        #     BIEO_labels.view(-1)
        # )
        # probability distribution over ts tag

        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score
        alpha = calculate_confidence(vec=p_y_x_ote_softmax)
        # transition score from ote tag to sentiment tag
        # a = self.transition_scores.unsqueeze(0)
        # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])
        ote2ts = torch.matmul(p_y_x_ote, self.transition_scores.to('cuda'))
        # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 7]), ote2ts) + torch.mul((1 - alpha),
                                                                                                         p_y_x_ts_softmax)
        if labels is not None: # training
            loss_ote = loss_fct(
                p_y_x_ote.view(-1, self.dim_ote_y),
                BIEO_labels.view(-1)
            )
            loss_ts = loss_fct(
                p_y_x_ts.view(-1, self.dim_ts_y),
                BIEOstance_labels.view(-1)
            )
            loss_i = loss_ote + loss_ts
            return loss_i
        else: # inference
            return torch.argmax(p_y_x_ote, dim=-1), torch.argmax(p_y_x_ts, dim=-1)



class Token_OTE2TS3(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_OTE2TS3, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 50
        self.dim_ote_y = 4
        self.dim_ts_y = 7
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.pool_s = nn.MaxPool1d(99, 1)
        self.pool_e = nn.MaxPool1d(99, 1)

        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'B': ['B-PRO', 'B-CON'],
                           'I': ['I-PRO', 'I-CON'],
                           'E': ['E-PRO', 'E-CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((4, 7))

        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
        ts_tag_vocab = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(1, 2 * self.dim_ote_h))
        self.dropout = nn.Dropout(0.1)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        if self.use_crf:
            self.crf = CRF(7, batch_first=self.batch_first)



    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, BIEO_labels=None, BIEOstance_labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.tokenbert.dropout(sequence_output)  # (B, L, H)
        #sequence_output, _ = self.rnn(sequence_output.view(sequence_output.shape[0], -1, 768))
        #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #p_y_x_ts = self.fc_ts(ote_hs)

        xs = self.conv_s(sequence_output.permute(0,2,1))
        xs = torch.relu(xs)
        xs = self.pool_s(xs)

        xe = self.conv_e(sequence_output.permute(0, 2, 1))
        xe = torch.relu(xe)
        xe = self.pool_e(xe)

        logits = self.tokenbert.classifier(sequence_output)
        if self.use_crf:
            if labels is not None: # training
                return -self.crf(logits, BIEOstance_labels, attention_mask.byte())
            else: # inference
                return self.crf.decode(logits, attention_mask.byte())
        else:
            if labels is not None: # training
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    p_y_x_ts.view(-1, self.num_labels),
                    BIEOstance_labels.view(-1)
                )
                return loss
            else: # inference
                return torch.argmax(p_y_x_ts, dim=2)


class Token_indicator_OTE2TS3(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_OTE2TS3, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 60
        self.dim_ote_y = 4
        self.dim_ts_y = 7
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)
        self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.pool_s = nn.MaxPool1d(99, 1)
        self.pool_e = nn.MaxPool1d(99, 1)

        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'B': ['B-PRO', 'B-CON'],
                           'I': ['I-PRO', 'I-CON'],
                           'E': ['E-PRO', 'E-CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((4, 7))

        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
        ts_tag_vocab = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.dropout = nn.Dropout(0.1)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(4, batch_first=self.batch_first)

        self.crf_ts = CRF(7, batch_first=self.batch_first)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, BIEO_labels=None, BIEO_stance_labels=None, stance_label=None, argument_label=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        cls_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)#只判断边界
        #
        ts_hs,_ = self.gru_ts(ote_hs)
        #边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                #ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                #print(h_tilde_t)
            #ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            #ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)
        #     if t==0:
        #         ts_hs_tildes=ts_hs_tilde.unsqueeze(0)
        #     else:
        p_y_x_ote = self.fc_ote(stm_lm_hs)
        #loss_fct = nn.CrossEntropyLoss()
        #p_y_x_ote = self.fc_ote(ote_hs)
        # loss_ote = loss_fct(
        #     p_y_x_ote.view(-1, self.dim_ote_y),
        #     BIEO_labels.view(-1)
        # )
        # probability distribution over ts tag

        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score
        alpha = calculate_confidence(vec=p_y_x_ote_softmax)
        # transition score from ote tag to sentiment tag
        # a = self.transition_scores.unsqueeze(0)
        # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])
        ote2ts = torch.matmul(p_y_x_ote, self.transition_scores.to('cuda'))
        # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 7]), ote2ts) + torch.mul((1 - alpha),
                                                                                                         p_y_x_ts_softmax)

        #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #p_y_x_ts = self.fc_ts(ote_hs)

        # xs = self.conv_s(sequence_output.permute(0,2,1))
        # xs = torch.relu(xs)
        # xs = self.pool_s(xs)
        #
        # xe = self.conv_e(sequence_output.permute(0, 2, 1))
        # xe = torch.relu(xe)
        # xe = self.pool_e(xe)
        if labels is not None:
            loss_ote = -self.crf_ote(p_y_x_ote, BIEO_labels, attention_mask.byte())
            loss_ts = -self.crf_ts(p_y_x_ts_tilde, BIEO_stance_labels, attention_mask.byte())
            return loss_ts + loss_ote
        else:  # inference
            return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()))

class Token_indicator_multitask(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 4
        self.dim_ts_y = 7
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)
        self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        self.pool_s = nn.MaxPool1d(99, 1)
        self.pool_e = nn.MaxPool1d(99, 1)

        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'B': ['B-PRO', 'B-CON'],
                           'I': ['I-PRO', 'I-CON'],
                           'E': ['E-PRO', 'E-CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((4, 7))

        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
        ts_tag_vocab = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(4, batch_first=self.batch_first)

        self.crf_ts = CRF(7, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype= torch.float32)
        #self.stance2argument[0]=[1, 0, 0]
        #self.stance2argument[1]=[0, 1, 1]


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, BIEO_labels=None, BIEO_stance_labels=None, stance_label=None, argument_label=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        cls_output = outputs[1]
        stance = self.stance(cls_output)
        argument = self.argument(cls_output)

        stance_softmax = F.softmax(stance, dim=1)
        argument_softmax = F.softmax(argument, dim=1)

        argument_t = torch.matmul(stance_softmax, self.stance2argument.to('cuda'))#stance到argument的转移概率
        cos = torch.cosine_similarity(argument_t-0.5, argument_softmax-0.5, dim=1)#计算相似度

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)#只判断边界
        #
        ts_hs,_ = self.gru_ts(ote_hs)
        #边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                #ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                #print(h_tilde_t)
            #ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            #ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)
        #     if t==0:
        #         ts_hs_tildes=ts_hs_tilde.unsqueeze(0)
        #     else:
        p_y_x_ote = self.fc_ote(stm_lm_hs)
        #loss_fct = nn.CrossEntropyLoss()
        #p_y_x_ote = self.fc_ote(ote_hs)
        # loss_ote = loss_fct(
        #     p_y_x_ote.view(-1, self.dim_ote_y),
        #     BIEO_labels.view(-1)
        # )
        # probability distribution over ts tag

        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score
        alpha = calculate_confidence(vec=p_y_x_ote_softmax)
        # transition score from ote tag to sentiment tag
        # a = self.transition_scores.unsqueeze(0)
        # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])
        ote2ts = torch.matmul(p_y_x_ote, self.transition_scores.to('cuda'))
        # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 7]), ote2ts) + torch.mul((1 - alpha),
                                                                                                         p_y_x_ts_softmax)

        #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #p_y_x_ts = self.fc_ts(ote_hs)

        # xs = self.conv_s(sequence_output.permute(0,2,1))
        # xs = torch.relu(xs)
        # xs = self.pool_s(xs)
        #
        # xe = self.conv_e(sequence_output.permute(0, 2, 1))
        # xe = torch.relu(xe)
        # xe = self.pool_e(xe)
        # stance = self.stance(cls_output)
        # argument = self.argument(cls_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            stance_loss = loss_fct(
                stance.view(-1, 3),
                stance_label.view(-1)
            )
            argument_loss = loss_fct(
                argument.view(-1, 2),
                argument_label.view(-1)
            )
            loss_ote = -self.crf_ote(p_y_x_ote, BIEO_labels, attention_mask.byte())
            loss_ts = -self.crf_ts(p_y_x_ts_tilde, BIEO_stance_labels, attention_mask.byte())
            return loss_ote + loss_ts + stance_loss + argument_loss - torch.mean(cos), (loss_ote, loss_ts, stance_loss, argument_loss, torch.mean(cos))
        else:  # inference
            return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance, dim=1), torch.argmax(argument, dim=1))

class self_attention_layer(nn.Module):
    def __init__(self, n_hidden):
        """
        Self-attention layer
        * n_hidden [int]: hidden layer number (equal to 2*n_hidden if bi-direction)
        """
        super(self_attention_layer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, 1, bias=False)
        )

    def init_weights(self):
        """
        Initialize all the weights and biases for this layer
        """
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.02)
                nn.init.uniform_(m.bias, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        """
        Forward calculation of the layer
        * inputs [tensor]: input tensor (batch_size * max_seq_len * n_hidden)
        * seq_len [tensor]: sequence length (batch_size,)
        - outputs [tensor]: attention output (batch_size * n_hidden)
        """
        if inputs.dim() != 3 :
            raise ValueError("! Wrong dimemsion of the inputs parameters.")

        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).contiguous().view(now_batch_size, 1, max_seq_len)
        exp = torch.exp(alpha)

        if mask is not None:
            # mask = get_mask(inputs, seq_len)
            # mask = mask.contiguous().view(now_batch_size, 1, max_seq_len)
            mask = mask.unsqueeze(1)
            exp = exp * mask.float()

        sum_exp = exp.sum(-1, True) + 1e-9
        softmax_exp = exp / sum_exp.expand_as(exp).contiguous().view(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(softmax_exp, inputs).squeeze(-2)
        return outputs


class self_attention_layer2(nn.Module):
    def __init__(self, n_hidden):
        """
        Self-attention layer
        * n_hidden [int]: hidden layer number (equal to 2*n_hidden if bi-direction)
        """
        super(self_attention_layer2, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_hidden, 1, bias=False)
        )

    def init_weights(self):
        """
        Initialize all the weights and biases for this layer
        """
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.02)
                nn.init.uniform_(m.bias, -0.02, 0.02)

    def forward(self, inputs, mask=None):
        """
        Forward calculation of the layer
        * inputs [tensor]: input tensor (batch_size * max_seq_len * n_hidden)
        * seq_len [tensor]: sequence length (batch_size,)
        - outputs [tensor]: attention output (batch_size * n_hidden)
        """
        if inputs.dim() != 3 :
            raise ValueError("! Wrong dimemsion of the inputs parameters.")

        now_batch_size, max_seq_len, _ = inputs.size()
        alpha = self.attention(inputs).contiguous().view(now_batch_size, 1, max_seq_len)
        exp = torch.sigmoid(alpha)
        #exp = torch.exp(alpha)

        if mask is not None:
            # mask = get_mask(inputs, seq_len)
            # mask = mask.contiguous().view(now_batch_size, 1, max_seq_len)
            mask = mask.unsqueeze(1)
            exp = exp * mask.float()

        #sum_exp = exp.sum(-1, True) + 1e-9
        #softmax_exp = exp / sum_exp.expand_as(exp).contiguous().view(now_batch_size, 1, max_seq_len)
        outputs = torch.bmm(exp, inputs).squeeze(-2)
        return outputs
class Token_indicator_multitask2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)
        #self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        # self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.pool_s = nn.MaxPool1d(99, 1)
        # self.pool_e = nn.MaxPool1d(99, 1)
        self.max_len=100
        self.gru_ote = nn.GRU(input_size=768,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        self.gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                          hidden_size=self.dim_ote_h,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)

        transition_path = {'I': ['PRO', 'CON'], 'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1 }
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2*self.max_len, 3*self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 3))
        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype= torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype= torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0],[ 0, 1., 1.]]), dtype= torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype= torch.float32)
        #self.mask = torch.tensor(np.array([[1., 1e-9, 1e-9, 1e-9, 1., 1.]]), dtype= torch.float32)
        #self.stance2argument[0]=[1, 0, 0]
        #self.stance2argument[1]=[0, 1, 1]
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)
        #self.attention3 = self_attention_layer(3)


    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask, argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]


        # outputs2 = self.bert(
        #     argument_input_ids,
        #     attention_mask=argument_attention_mask,
        #     token_type_ids=argument_token_type_ids
        # )
        #stance_cls_output = outputs[1]
        #argument_cls_output = outputs[1]

        stance_pooled_output = self.attention(outputs[0], attention_mask)
        stance_pooled_output = self.dropout(stance_pooled_output)

        #stance = self.stance(stance_cls_output)
        stance = self.stance(stance_pooled_output)

        stance_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)#只判断边界
        #
        ts_hs,_ = self.gru_ts(ote_hs)
        #边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                #ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                #a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                #print(h_tilde_t)
            #ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            #ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)

        #sequence_stance_output = self.attention2(stm_lm_ts, attention_mask)
        #sequence_stance = self.sequence_stance(sequence_stance_output)
        #a=sequence_stance.unsqueeze(1)
        #b=torch.transpose(stance.unsqueeze(1), 1, 2)
        #c= torch.matmul(sequence_stance.unsqueeze(1), self.W_stance_gate.to('cuda'))
        #stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        #stacne_gat = stacne_gat.squeeze(-1)
        #stance_log = stacne_gat * F.softmax(stance, dim=1) + (1 - stacne_gat) * F.softmax(sequence_stance, dim=1)
        #stance_log = 0.5 * F.softmax(stance, dim=1) + 0.5 * F.softmax(sequence_stance, dim=1)

        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score

        #p_y_x_ote_softmax2_unsequence = p_y_x_ote_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(1)
        #p_y_x_ts_softmax_unsequence = p_y_x_ts_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(2)
        #alpha = torch.matmul(torch.matmul(p_y_x_ote_softmax2_unsequence, self.W_gate2.to('cuda')), p_y_x_ts_softmax_unsequence)
        #alpha = calculate_confidence(vec=p_y_x_ote_softmax)



        #alpha = self.attention3(p_y_x_ote_softmax, attention_mask)
        # transition score from ote tag to sentiment tag
        # a = self.transition_scores.unsqueeze(0)
        # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])

#ote2ts
        #p_y_x_ote_softmax, [32,100,2],stance_softmax,[32,3]
        # IO_labelss = torch.zeros(sequence_output.shape[0]*self.max_len, 2).to('cuda')
        # IO_labelssss = IO_labelss.scatter_(1, IO_labels.contiguous().view(-1,1), 1)
        # p_y_x_ote_softmax_unsequence = IO_labelssss.unsqueeze(1)
        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1,1,2)
        # stancess = torch.zeros(sequence_output.shape[0], 3).to('cuda').scatter_(1, stance_label.contiguous().view(-1, 1), 1)
        # stance_softmax_unsequence = stancess.unsqueeze(1)
        # stance_softmax_unsequence = stance_softmax_unsequence.expand(sequence_output.shape[0],self.max_len,3).contiguous().view(-1,1,3)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0],self.max_len,3).contiguous().view(-1,1,3)
        #ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,6)*self.mask.to('cuda').expand([sequence_output.shape[0]*self.max_len, 6])
        #ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,6)*self.mask.to('cuda').expand([sequence_output.shape[0]*self.max_len, 6])
        #ote2ts2 = ote2ts2.contiguous().view(-1,2,3)
        #ote2ts = torch.sum(ote2ts2, dim=1).contiguous().view(-1,self.max_len,3)

#复制0的转移向量
        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1,1,3)
#拼接
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
#转移概率
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa*self.mask2.to('cuda')).contiguous().view(-1,self.max_len,3)

        #p_y_x_ote_softmax_unsequencessss = p_y_x_ote_softmax_unsequence.contiguous().view(-1,self.max_len,2).cpu().tolist()
        #stance_softmax_unsequencessss = stance_softmax_unsequence.contiguous().view(-1,self.max_len,3).cpu().tolist()
        #ote2ts2sss =torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,self.max_len, 2,3).cpu().tolist()
        #p_y_x_ote_softmax_unsequence2 = IO_labelssss.unsqueeze(1)
        #a=aaaa*self.mask2.to('cuda')
        #b=torch.mul(aaaa,self.mask2.to('cuda'))
        #c=torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa*self.mask2.to('cuda'))
        #ote2ts2a = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa*self.mask2.to('cuda')).contiguous().view(-1,self.max_len,3)
            #.expand([sequence_output.shape[0]*self.max_len, 6])
        #ote2ts2a = ote2ts2a
        #ote2ts2assss = ote2ts2a.cpu().tolist()
        #aa = aaaa.cpu().tolist()
        #ote2tsssss = ote2ts.cpu().tolist()
        #labelsssss = labels.cpu().tolist()

        #ote2ts = F.softmax(torch.sum(ote2ts2, dim=1).contiguous().view(-1,self.max_len,3), dim=-1)
        #ote2ts = torch.matmul(p_y_x_ote_softmax, self.transition_scores.to('cuda'))
        # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        #p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 3]), ote2ts) + torch.mul((1 - alpha), p_y_x_ts_softmax)


        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        #sequence_stance = self.sequence_stance(sequence_stance_output)
        #stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance_output.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        #stacne_gat = stacne_gat.squeeze(-1)
        #再加一个交叉熵验证
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)
        #stance_log = stacne_gat * F.softmax(stance, dim=1) + (1 - stacne_gat) * F.softmax(sequence_stance_output, dim=1)
        #stance_log = 0.5 * F.softmax(stance, dim=1) + 0.5 * F.softmax(p_y_x_ts_tilde, dim=1)
        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))#stance到argument的转移概率

        #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #p_y_x_ts = self.fc_ts(ote_hs)

        # xs = self.conv_s(sequence_output.permute(0,2,1))
        # xs = torch.relu(xs)
        # xs = self.pool_s(xs)
        #
        # xe = self.conv_e(sequence_output.permute(0, 2, 1))
        # xe = torch.relu(xe)
        # xe = self.pool_e(xe)
        # stance = self.stance(cls_output)
        # argument = self.argument(cls_output)
        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss1 = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )

                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                #ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction = 'token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction = 'token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction = 'token_mean')
                return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss , (loss_ote, loss_ts1, loss_ts2, stance_loss1, stance_loss2, argument_t_loss)
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1), torch.argmax(sequence_stance_output, dim=1))
        else:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_loss = loss_fct(
                    argument.view(-1, 2),
                    argument_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                loss_ote = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )

                loss_ts = loss_fct(
                    p_y_x_ts_tilde.view(-1, 3),
                    labels.view(-1)
                )
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote + loss_ts + stance_loss + argument_loss + argument_t_loss, (loss_ote, loss_ts, stance_loss, argument_loss, argument_t_loss)
            else:  # inference
                return (torch.argmax(p_y_x_ote, dim=2),torch.argmax(p_y_x_ts_tilde, dim=2), torch.argmax(stance, dim=1),)
