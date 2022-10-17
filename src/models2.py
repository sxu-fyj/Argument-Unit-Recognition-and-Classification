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
class Token_indicator_multitask_gat(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_gat, self).__init__()
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
        self.dropout = nn.Dropout(0.5)

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
        self.attention3 = self_attention_layer(3)


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
        stance_cls_output = outputs[1]
        #argument_cls_output = outputs[1]

        stance_pooled_output = self.attention(outputs[0], attention_mask)
        stance_pooled_output = self.dropout(stance_pooled_output)

        stance = self.stance(stance_cls_output)
        #stance = self.stance(stance_pooled_output)

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


        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax求和
        #门控
        #alpha = self.attention3(p_y_x_ote_softmax, attention_mask)
        p_y_x_ote_softmax2_unsequence = p_y_x_ote_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(1)
        p_y_x_ts_softmax_unsequence = p_y_x_ts_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(2)
        alpha = torch.sigmoid(torch.matmul(torch.matmul(p_y_x_ote_softmax2_unsequence, self.W_gate2.to('cuda')), p_y_x_ts_softmax_unsequence))
        #a=alpha *ote2ts
        #b=(1 - alpha)*p_y_x_ts_softmax
        p_y_x_ts_tilde = torch.mul(alpha, ote2ts) + torch.mul((1 - alpha), p_y_x_ts_softmax)


        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        #sequence_stance = self.sequence_stance(sequence_stance_output)
        #stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance_output.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        #stacne_gat = stacne_gat.squeeze(-1)
        #再加一个交叉熵验证

        stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance_output.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        stacne_gat = stacne_gat.squeeze(-1)
        stance_log = stacne_gat * F.softmax(stance, dim=1) + (1 - stacne_gat) * F.softmax(sequence_stance_output, dim=1)#门控

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)#加和



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
        self.dropout = nn.Dropout(0.5)

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
        stance_cls_output = outputs[1]
        #argument_cls_output = outputs[1]

        stance_pooled_output = self.attention(outputs[0], attention_mask)
        stance_pooled_output = self.dropout(stance_pooled_output)

        stance = self.stance(stance_cls_output)
        #stance = self.stance(stance_pooled_output)

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

#DRNN实现






class Token_indicator_multitask_addx2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_addx2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
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

        self.drnn = nn.GRU(input_size=2 * self.dim_ote_h,
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

        self.conv_s = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=2, stride=1)
        self.conv_e = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=2, stride=1)
        self.pool_s = nn.MaxPool1d(99, 1)
        self.pool_e = nn.MaxPool1d(99, 1)

        self.pool_s = nn.MaxPool1d(99, 1)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2*self.max_len, 3*self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 3))
        self.dropout = nn.Dropout(0.1)

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
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)
        self.attentionx2_2 = self_attention_layer2(3)
        self.attentionx3_2 = self_attention_layer2(3)
        self.roo = nn.GRUCell(3, 3)
        self.Wdrnn = nn.Linear(in_features=2 * self.dim_ote_h, out_features=2 * self.dim_ote_h, bias=True)

        self.num_k=2
        #self.pad = nn.ZeroPad2d(padding=(0, 0, 0, self.num_k - 1))
    def DRNN(self, input, attention_mask):
        a=torch.zeros(attention_mask.shape[0],1)
        b = F.pad(a, [0, attention_mask.shape[1]-1, 0, 0], value=1 ).unsqueeze(2)
        bbb = b.cpu().tolist()
        attention_mask2 =  F.pad(a, [0, attention_mask.shape[1]-1, 0, 0], value=1).cuda() * attention_mask
        #attention_maskc2 =attention_mask2.cpu().tolist()
        #attention_maskc=attention_mask.cpu().tolist()
        input = input * attention_mask2.unsqueeze(2)
        self.hidden = []
        #print(input.shape)
        #num_k窗口
        start = 0
        end = start + self.num_k
        while end <= (input.shape[1]):
            input_k = input[:, start:end, :]
            #print(input_k.shape)
            enconder_outputs, state = self.drnn(input_k)    #每个序列的输出, 最终状态
            state1 = torch.cat((state[0],state[1]), 1)
            state_dropout = self.dropout(state1)
            mlp_output = torch.relu(self.Wdrnn(state_dropout)).unsqueeze(1)
            self.hidden.append(mlp_output)
            start += 1
            end += 1
        hidden_concat = torch.cat(self.hidden, 1)
        return hidden_concat
    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask, argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
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

        #stance 分类
        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_softmax = F.softmax(stance, dim=1)

        # 只判断边界
        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)

        #边界的基础上判断情感
        ts_hs,_ = self.gru_ts(ote_hs)
        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        gts=[]

        drnn_out=self.DRNN(ote_hs, attention_mask)
        start_p = self.drnn_start_linear(drnn_out)
        end_p = self.drnn_end_linear(drnn_out)
        # xs = self.conv_s(ote_hs.permute(0, 2, 1))
        # xs = torch.relu(xs)
        # xs = self.pool_s(xs)
        #
        # xe = self.conv_e(ote_hs.permute(0, 2, 1))
        # xe = torch.relu(xe)
        # xe = self.pool_e(xe)

        #bert-e2e中有gru的实现
        #通过判断边界的一致性来帮助
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                #ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                #如果前一个状态和当前状态相似，则延续立场标签，否则立场标签不一样。
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda'))+torch.mm(h_tilde_tm1, self.W_gate.to('cuda')))
                #gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                #保存gt的值，观察每次在断点处gt的变化，多个二分类，此门控的值用来判断边界
                #a= (1 - gt)
                gts.append(gt)
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

        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        #h_0 = Variable(torch.zeros(sequence_output.shape[0], 3)).cuda()
        #h_t = h_0  # 立场向量，初始立场向量为全0




        #通过转移概率计算最终的label
        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1,1,2)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0],self.max_len,3).contiguous().view(-1,1,3)
        #复制标签O的转移向量
        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1,1,3)
        #拼接
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        #转移概率
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa*self.mask2.to('cuda')).contiguous().view(-1,self.max_len,3)

        #2种方式获得的label权重加和
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax

        #通过序列标注的标签得到立场标签
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        #h_t = self.roo(stance, h_t)  # blog.csdn.net/duan_zhihua/article/details/82763119，就是GRU的其中一个状态。
        #h_t = self.roo(sequence_stance_output, h_t)  # blog.csdn.net/duan_zhihua/article/details/82763119，就是GRU的其中一个状态。

        #h_t = F.tanh(h_t)  # maybe useful
        #第二层

        # p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # stance_softmax_unsequence2 = h_t.unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1, 1, 3)
        # # 复制标签O的转移向量
        # transs2 = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
        #     -1, 1, 3)
        # # 拼接
        # aaaa2 = torch.cat((transs2, stance_softmax_unsequence2), 1).contiguous().view(-1, 2, 3)
        # # 转移概率
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa2 * self.mask2.to('cuda')).contiguous().view(-1, self.max_len, 3)
        # p_y_x_ts_tilde2 = ote2ts2 + p_y_x_ts_tilde
        #
        # sequence_stance_output2 = self.attentionx2_2(p_y_x_ts_tilde2, attention_mask)
        #
        # #stance_log2 = F.softmax(stance_log, dim=1) + F.softmax(sequence_stance_output2, dim=1)
        #
        # h_t = self.roo(sequence_stance_output2, h_t)  # blog.csdn.net/duan_zhihua/article/details/82763119，就是GRU的其中一个状态。
        #
        # #h_t = F.tanh(h_t)
        # #第三层
        # p_y_x_ote_softmax_unsequence3 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # stance_softmax_unsequence3 = h_t.unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1, 1, 3)
        # # 复制标签O的转移向量
        # transs3 = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1, 1, 3)
        # # 拼接
        # aaaa3 = torch.cat((transs3, stance_softmax_unsequence3), 1).contiguous().view(-1, 2, 3)
        # # 转移概率
        # ote2ts3 = torch.matmul(p_y_x_ote_softmax_unsequence3, aaaa3 * self.mask2.to('cuda')).contiguous().view(-1,
        #                                                                                                        self.max_len,
        #                                                                                                        3)
        # p_y_x_ts_tilde3 = ote2ts3 + p_y_x_ts_tilde2
        #
        # sequence_stance_output3 = self.attentionx3_2(p_y_x_ts_tilde3, attention_mask)
        # #stance_log3 = F.softmax(stance_log2, dim=1) + F.softmax(sequence_stance_output3, dim=1)
        #
        # stance_log3 = self.roo(sequence_stance_output2, h_t)  # blog.csdn.net/duan_zhihua/article/details/82763119，就是GRU的其中一个状态。

        #stance限制
        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))#stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                #ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction = 'token_mean')
                loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction = 'token_mean')
                return loss_ote + loss_ts + start_loss + end_loss + stance_loss + argument_t_loss , (loss_ote, loss_ts, stance, stance_loss, start_loss, end_loss, argument_t_loss, argument_t_loss, argument_t_loss, argument_t_loss)
                #return loss_ote + loss_ts1 + loss_ts2 + loss_tsx2_1 + loss_tsx2_2 + stance_loss1 + stance_loss2 + stance_lossx2_1 + stance_lossx2_2 + argument_t_loss , (loss_ote, loss_ts1, loss_ts2, loss_tsx2_1, loss_tsx2_2, stance_loss1, stance_loss2, stance_lossx2_1, stance_lossx2_2, argument_t_loss)
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log3, dim=1),
                        torch.argmax(stance_log3, dim=1), torch.argmax(stance_log3, dim=1), torch.argmax(stance_log3, dim=1))


        # if self.use_crf:
        #     if labels is not None:
        #         loss_fct = nn.CrossEntropyLoss()
        #         start_loss = loss_fct(
        #             start_p.view(-1, 2),
        #             start_label.view(-1)
        #         )
        #         end_loss = loss_fct(
        #             end_p.view(-1, 2),
        #             end_label.view(-1)
        #         )
        #         stance_loss1 = loss_fct(
        #             stance_log3.view(-1, 3),
        #             stance_label.view(-1)
        #         )
        #         stance_loss2 = loss_fct(
        #             stance_log3.view(-1, 3),
        #             stance_label.view(-1)
        #         )
        #
        #         stance_lossx2_1 = loss_fct(
        #             stance_log3.view(-1, 3),
        #             stance_label.view(-1)
        #         )
        #         stance_lossx2_2 = loss_fct(
        #             stance_log3.view(-1, 3),
        #             stance_label.view(-1)
        #         )
        #
        #         argument_t_loss = loss_fct(
        #             argument_t.view(-1, 2),
        #             argument_label.view(-1)
        #         )
        #
        #         #ote2ts
        #         loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction = 'token_mean')
        #         loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction = 'token_mean')
        #         loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction = 'token_mean')
        #         loss_tsx2_2 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction = 'token_mean')
        #         loss_tsx2_1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction = 'token_mean')
        #         loss_tsx3_2 = -self.crf_ts(ote2ts3, labels, attention_mask.byte(), reduction='token_mean')
        #         loss_tsx3_1 = -self.crf_ts(p_y_x_ts_tilde3, labels, attention_mask.byte(), reduction='token_mean')
        #         return loss_ote + loss_tsx3_1 + loss_tsx3_2 + stance_lossx2_1 + argument_t_loss , (loss_ote, loss_ts1, loss_ts2, loss_tsx3_1, loss_tsx3_2, stance_loss1, stance_loss2, stance_lossx2_1, stance_lossx2_2, argument_t_loss)
        #         #return loss_ote + loss_ts1 + loss_ts2 + loss_tsx2_1 + loss_tsx2_2 + stance_loss1 + stance_loss2 + stance_lossx2_1 + stance_lossx2_2 + argument_t_loss , (loss_ote, loss_ts1, loss_ts2, loss_tsx2_1, loss_tsx2_2, stance_loss1, stance_loss2, stance_lossx2_1, stance_lossx2_2, argument_t_loss)
        #         #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
        #     else:  # inference
        #         return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), self.crf_ts.decode(ote2ts, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()), self.crf_ts.decode(ote2ts2, attention_mask.byte()), torch.argmax(stance_log3, dim=1), torch.argmax(stance_log3, dim=1), torch.argmax(stance_log3, dim=1), torch.argmax(stance_log3, dim=1))
        # else:
        #     if labels is not None:
        #         loss_fct = nn.CrossEntropyLoss()
        #         stance_loss = loss_fct(
        #             stance_log.view(-1, 3),
        #             stance_label.view(-1)
        #         )
        #         argument_loss = loss_fct(
        #             argument.view(-1, 2),
        #             argument_label.view(-1)
        #         )
        #         argument_t_loss = loss_fct(
        #             argument_t.view(-1, 2),
        #             argument_label.view(-1)
        #         )
        #
        #         loss_ote = loss_fct(
        #             p_y_x_ote.view(-1, 2),
        #             IO_labels.view(-1)
        #         )
        #
        #         loss_ts = loss_fct(
        #             p_y_x_ts_tilde.view(-1, 3),
        #             labels.view(-1)
        #         )
        #         #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
        #         #loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
        #         return loss_ote + loss_ts + stance_loss + argument_loss + argument_t_loss, (loss_ote, loss_ts, stance_loss, argument_loss, argument_t_loss)
        #     else:  # inference
        #         return (torch.argmax(p_y_x_ote, dim=2),torch.argmax(p_y_x_ts_tilde, dim=2), torch.argmax(stance, dim=1),)


class Token_indicator_multitask_se(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
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

        self.drnn = nn.GRU(input_size=2 * self.dim_ote_h,
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

        self.conv_s = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=2, stride=1)
        self.conv_e = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=2, stride=1)
        self.pool_s = nn.MaxPool1d(99, 1)
        self.pool_e = nn.MaxPool1d(99, 1)

        self.pool_s = nn.MaxPool1d(99, 1)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
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
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)
        self.attentionx2_2 = self_attention_layer2(3)
        self.attentionx3_2 = self_attention_layer2(3)
        self.roo = nn.GRUCell(3, 3)
        self.Wdrnn = nn.Linear(in_features=2 * self.dim_ote_h, out_features=2 * self.dim_ote_h, bias=True)

        self.num_k=2
        #self.pad = nn.ZeroPad2d(padding=(0, 0, 0, self.num_k - 1))
    def DRNN(self, input, attention_mask):
        a=torch.zeros(attention_mask.shape[0],1)
        b = F.pad(a, [0, attention_mask.shape[1]-1, 0, 0], value=1 ).unsqueeze(2)
        bbb = b.cpu().tolist()
        attention_mask2 =  F.pad(a, [0, attention_mask.shape[1]-1, 0, 0], value=1).cuda() * attention_mask
        #attention_maskc2 =attention_mask2.cpu().tolist()
        #attention_maskc=attention_mask.cpu().tolist()
        input = input * attention_mask2.unsqueeze(2)
        self.hidden = []
        #print(input.shape)
        #num_k窗口
        start = 0
        end = start + self.num_k
        while end <= (input.shape[1]):
            input_k = input[:, start:end, :]
            #print(input_k.shape)
            enconder_outputs, state = self.drnn(input_k)    #每个序列的输出, 最终状态
            state1 = torch.cat((state[0],state[1]), 1)
            state_dropout = self.dropout(state1)
            mlp_output = torch.relu(self.Wdrnn(state_dropout)).unsqueeze(1)
            self.hidden.append(mlp_output)
            start += 1
            end += 1
        hidden_concat = torch.cat(self.hidden, 1)
        return hidden_concat
    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask, argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
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

        #stance 分类
        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_softmax = F.softmax(stance, dim=1)

        # 只判断边界
        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)

        #边界的基础上判断情感
        ts_hs,_ = self.gru_ts(ote_hs)
        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object

        drnn_out=self.DRNN(ote_hs, attention_mask)
        start_p = self.drnn_start_linear(drnn_out)
        end_p = self.drnn_end_linear(drnn_out)
        # xs = self.conv_s(ote_hs.permute(0, 2, 1))
        # xs = torch.relu(xs)
        # xs = self.pool_s(xs)
        #
        # xe = self.conv_e(ote_hs.permute(0, 2, 1))
        # xe = torch.relu(xe)
        # xe = self.pool_e(xe)

        #bert-e2e中有gru的实现
        #通过判断边界的一致性来帮助
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                #ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                #如果前一个状态和当前状态相似，则延续立场标签，否则立场标签不一样。
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda'))+torch.mm(h_tilde_tm1, self.W_gate.to('cuda')))
                #gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                #保存gt的值，观察每次在断点处gt的变化，多个二分类，此门控的值用来判断边界
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

        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        #h_0 = Variable(torch.zeros(sequence_output.shape[0], 3)).cuda()
        #h_t = h_0  # 立场向量，初始立场向量为全0




        #通过转移概率计算最终的label
        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1,1,2)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0],self.max_len,3).contiguous().view(-1,1,3)
        #复制标签O的转移向量
        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(-1,1,3)
        #拼接
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        #转移概率
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa*self.mask2.to('cuda')).contiguous().view(-1,self.max_len,3)

        #2种方式获得的label权重加和
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax

        #通过序列标注的标签得到立场标签
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))#stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )


                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )



                #ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction = 'token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction = 'token_mean')
                loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction = 'token_mean')
                return loss_ote + loss_ts + loss_ts2 + stance_loss + stance_loss2 + argument_t_loss , (loss_ote, loss_ts, start_loss, end_loss, stance_loss, argument_t_loss)
                #return loss_ote + loss_ts + start_loss + end_loss + stance_loss + argument_t_loss , (loss_ote, loss_ts, start_loss, end_loss, stance_loss, argument_t_loss)
                #return loss_ote + loss_ts1 + loss_ts2 + loss_tsx2_1 + loss_tsx2_2 + stance_loss1 + stance_loss2 + stance_lossx2_1 + stance_lossx2_2 + argument_t_loss , (loss_ote, loss_ts1, loss_ts2, loss_tsx2_1, loss_tsx2_2, stance_loss1, stance_loss2, stance_lossx2_1, stance_lossx2_2, argument_t_loss)
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(start_p, dim=1), torch.argmax(end_p, dim=1), torch.argmax(stance_log, dim=1))

class Token_indicator_multitask_se2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        # self.batch_size=batch_size
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
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        # self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.pool_s = nn.MaxPool1d(99, 1)
        # self.pool_e = nn.MaxPool1d(99, 1)
        self.max_len = 100
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

        self.drnn = nn.GRU(input_size=2 * self.dim_ote_h,
                           hidden_size=self.dim_ote_h,
                           num_layers=1,
                           bias=True,
                           dropout=0,
                           batch_first=True,
                           bidirectional=True)
        transition_path = {'I': ['PRO', 'CON'], 'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2 * self.max_len, 3 * self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 3))
        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        # self.mask = torch.tensor(np.array([[1., 1e-9, 1e-9, 1e-9, 1., 1.]]), dtype= torch.float32)
        # self.stance2argument[0]=[1, 0, 0]
        # self.stance2argument[1]=[0, 1, 1]
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)
        # self.attention3 = self_attention_layer(3)
        self.Wdrnn = nn.Linear(in_features=2 * self.dim_ote_h, out_features=2 * self.dim_ote_h, bias=True)

        self.num_k = 2
        # self.pad = nn.ZeroPad2d(padding=(0, 0, 0, self.num_k - 1))

    def DRNN(self, input, attention_mask):
        a = torch.zeros(attention_mask.shape[0], 1)
        b = F.pad(a, [0, attention_mask.shape[1] - 1, 0, 0], value=1).unsqueeze(2)
        bbb = b.cpu().tolist()
        attention_mask2 = F.pad(a, [0, attention_mask.shape[1] - 1, 0, 0], value=1).cuda() * attention_mask
        # attention_maskc2 =attention_mask2.cpu().tolist()
        # attention_maskc=attention_mask.cpu().tolist()
        input = input * attention_mask2.unsqueeze(2)
        self.hidden = []
        # print(input.shape)
        # num_k窗口
        start = 0
        end = start + self.num_k
        while end <= (input.shape[1]):
            input_k = input[:, start:end, :]
            # print(input_k.shape)
            enconder_outputs, state = self.drnn(input_k)  # 每个序列的输出, 最终状态
            state1 = torch.cat((state[0], state[1]), 1)
            state_dropout = self.dropout(state1)
            mlp_output = torch.relu(self.Wdrnn(state_dropout)).unsqueeze(1)
            self.hidden.append(mlp_output)
            start += 1
            end += 1
        hidden_concat = torch.cat(self.hidden, 1)
        return hidden_concat
    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.tokenbert.bert(
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)

        drnn_out = self.DRNN(ote_hs, attention_mask)
        start_p = self.drnn_start_linear(drnn_out)
        end_p = self.drnn_end_linear(drnn_out)
        # sequence_stance_output = self.attention2(stm_lm_ts, attention_mask)
        # sequence_stance = self.sequence_stance(sequence_stance_output)
        # a=sequence_stance.unsqueeze(1)
        # b=torch.transpose(stance.unsqueeze(1), 1, 2)
        # c= torch.matmul(sequence_stance.unsqueeze(1), self.W_stance_gate.to('cuda'))
        # stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        # stacne_gat = stacne_gat.squeeze(-1)
        # stance_log = stacne_gat * F.softmax(stance, dim=1) + (1 - stacne_gat) * F.softmax(sequence_stance, dim=1)
        # stance_log = 0.5 * F.softmax(stance, dim=1) + 0.5 * F.softmax(sequence_stance, dim=1)

        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        # normalized the score

        # p_y_x_ote_softmax2_unsequence = p_y_x_ote_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(1)
        # p_y_x_ts_softmax_unsequence = p_y_x_ts_softmax.contiguous().view(sequence_output.shape[0], -1).unsqueeze(2)
        # alpha = torch.matmul(torch.matmul(p_y_x_ote_softmax2_unsequence, self.W_gate2.to('cuda')), p_y_x_ts_softmax_unsequence)
        # alpha = calculate_confidence(vec=p_y_x_ote_softmax)

        # alpha = self.attention3(p_y_x_ote_softmax, attention_mask)
        # transition score from ote tag to sentiment tag
        # a = self.transition_scores.unsqueeze(0)
        # b= a.expand([32,self.transition_scores.shape[0],self.transition_scores.shape[1]])

        # ote2ts
        # p_y_x_ote_softmax, [32,100,2],stance_softmax,[32,3]
        # IO_labelss = torch.zeros(sequence_output.shape[0]*self.max_len, 2).to('cuda')
        # IO_labelssss = IO_labelss.scatter_(1, IO_labels.contiguous().view(-1,1), 1)
        # p_y_x_ote_softmax_unsequence = IO_labelssss.unsqueeze(1)
        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # stancess = torch.zeros(sequence_output.shape[0], 3).to('cuda').scatter_(1, stance_label.contiguous().view(-1, 1), 1)
        # stance_softmax_unsequence = stancess.unsqueeze(1)
        # stance_softmax_unsequence = stance_softmax_unsequence.expand(sequence_output.shape[0],self.max_len,3).contiguous().view(-1,1,3)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,6)*self.mask.to('cuda').expand([sequence_output.shape[0]*self.max_len, 6])
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,6)*self.mask.to('cuda').expand([sequence_output.shape[0]*self.max_len, 6])
        # ote2ts2 = ote2ts2.contiguous().view(-1,2,3)
        # ote2ts = torch.sum(ote2ts2, dim=1).contiguous().view(-1,self.max_len,3)

        # 复制0的转移向量
        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        # 拼接
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        # 转移概率
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        # p_y_x_ote_softmax_unsequencessss = p_y_x_ote_softmax_unsequence.contiguous().view(-1,self.max_len,2).cpu().tolist()
        # stance_softmax_unsequencessss = stance_softmax_unsequence.contiguous().view(-1,self.max_len,3).cpu().tolist()
        # ote2ts2sss =torch.matmul(p_y_x_ote_softmax_unsequence, stance_softmax_unsequence).contiguous().view(-1,self.max_len, 2,3).cpu().tolist()
        # p_y_x_ote_softmax_unsequence2 = IO_labelssss.unsqueeze(1)
        # a=aaaa*self.mask2.to('cuda')
        # b=torch.mul(aaaa,self.mask2.to('cuda'))
        # c=torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa*self.mask2.to('cuda'))
        # ote2ts2a = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa*self.mask2.to('cuda')).contiguous().view(-1,self.max_len,3)
        # .expand([sequence_output.shape[0]*self.max_len, 6])
        # ote2ts2a = ote2ts2a
        # ote2ts2assss = ote2ts2a.cpu().tolist()
        # aa = aaaa.cpu().tolist()
        # ote2tsssss = ote2ts.cpu().tolist()
        # labelsssss = labels.cpu().tolist()

        # ote2ts = F.softmax(torch.sum(ote2ts2, dim=1).contiguous().view(-1,self.max_len,3), dim=-1)
        # ote2ts = torch.matmul(p_y_x_ote_softmax, self.transition_scores.to('cuda'))
        # ote2ts2 = torch.matmul(p_y_x_ote, b.to('cuda'))
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        # p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 100, 3]), ote2ts) + torch.mul((1 - alpha), p_y_x_ts_softmax)

        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # sequence_stance = self.sequence_stance(sequence_stance_output)
        # stacne_gat = torch.sigmoid(torch.matmul(torch.matmul(sequence_stance_output.unsqueeze(1), self.W_stance_gate.to('cuda')), torch.transpose(stance.unsqueeze(1), 1, 2)))
        # stacne_gat = stacne_gat.squeeze(-1)
        # 再加一个交叉熵验证
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)
        # stance_log = stacne_gat * F.softmax(stance, dim=1) + (1 - stacne_gat) * F.softmax(sequence_stance_output, dim=1)
        # stance_log = 0.5 * F.softmax(stance, dim=1) + 0.5 * F.softmax(p_y_x_ts_tilde, dim=1)
        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        # ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        # p_y_x_ts = self.fc_ts(ote_hs)

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
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
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

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1 + stance_loss1 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss1, stance_loss2, argument_t_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))
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
                # loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote + loss_ts + stance_loss + argument_loss + argument_t_loss, (
                loss_ote, loss_ts, stance_loss, argument_loss, argument_t_loss)
            else:  # inference
                return (
                torch.argmax(p_y_x_ote, dim=2), torch.argmax(p_y_x_ts_tilde, dim=2), torch.argmax(stance, dim=1),)


class Token_indicator_multitask3(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask3, self).__init__()
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
        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
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

class Token_indicator_multitask_se3(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se3, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        # self.batch_size=batch_size
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
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        # self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.pool_s = nn.MaxPool1d(99, 1)
        # self.pool_e = nn.MaxPool1d(99, 1)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2 * self.max_len, 3 * self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 3))
        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.tokenbert.bert(
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界

        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax

        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # 再加一个交叉熵验证
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)
        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
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

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss1, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se4(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se4, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        # self.batch_size=batch_size
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
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        # self.conv_s = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.conv_e = nn.Conv1d(in_channels=768, out_channels=100, kernel_size=2, stride=1)
        # self.pool_s = nn.MaxPool1d(99, 1)
        # self.pool_e = nn.MaxPool1d(99, 1)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2 * self.max_len, 3 * self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 3))
        self.dropout = nn.Dropout(0.5)

        self.stance = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.tokenbert.bert(
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界

        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)

        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax

        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # 再加一个交叉熵验证
        stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)
        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
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

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + stance_loss1 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + stance_loss1  + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss1, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))


class Token_indicator_multitask_se_1020(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1020, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate2 = torch.nn.init.xavier_uniform_(torch.empty(2 * self.max_len, 3 * self.max_len))
        self.W_stance_gate = torch.nn.init.xavier_uniform_(torch.empty(3, 1))
        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax

        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                # stance_loss2 = loss_fct(
                #     sequence_stance_output.view(-1, 3),
                #     stance_label.view(-1)
                # )

                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + stance_loss1 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + stance_loss1  + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1220_2(nn.Module):#全加
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1220_2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 1))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        self.bias = nn.Parameter(torch.FloatTensor(100,3))
        nn.init.xavier_uniform_(self.bias.data, gain=1)
        self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax
        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1220_3(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1220_3, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 100))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        self.bias = nn.Parameter(torch.FloatTensor(100,3))
        nn.init.xavier_uniform_(self.bias.data, gain=1)
        self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1221_1(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1221_1, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 100))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(100))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))
        # self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        # nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        # self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        # self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        # self.bias = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias.data, gain=1)
        # self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1)
        ts_cat1=torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1).contiguous().view(-1, 2, 300)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(ts_cat1, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
        alpha_ts1 = F.softmax(alpha_ts1.sum(1, keepdim=True), dim=2)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, ts_cat1)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, 100, 3)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)
        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        ts_cat2_a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1), ote2ts2.unsqueeze(1)), 1)
        ts_cat2 = ts_cat2_a.contiguous().view(-1, 3, 300)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(ts_cat2, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(1, 2)
        alpha_ts2 = F.softmax(alpha_ts2.sum(1, keepdim=True), dim=2)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, ts_cat2)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, 100, 3)


        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts4 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1 + loss_ts2 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1221_2(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1221_2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 100))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(100))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))
        # self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        # nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        # self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        # self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        # self.bias = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias.data, gain=1)
        # self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.5)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        # a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1)
        # ts_cat1=torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1).contiguous().view(-1, 2, 300)
        # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(ts_cat1, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
        # alpha_ts1 = F.softmax(alpha_ts1.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde = torch.matmul(alpha_ts1, ts_cat1)
        # p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, 100, 3)
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)
        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        # p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        # stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        #                                                                    3).contiguous().view(-1, 1, 3)
        #
        # transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
        #     -1, 1, 3)
        # aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
        #                                                                                                     self.max_len,
        #                                                                                                     3)
        # # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)
        #
        # #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # ts_cat2_a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1), ote2ts2.unsqueeze(1)), 1)
        # ts_cat2 = ts_cat2_a.contiguous().view(-1, 3, 300)
        # alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(ts_cat2, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(1, 2)
        # alpha_ts2 = F.softmax(alpha_ts2.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, ts_cat2)
        # p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, 100, 3)


        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4, (
                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1222_1(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1222_1, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 100))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(100))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))
        # self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        # nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        # self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        # self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        # self.bias = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias.data, gain=1)
        # self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.5)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2, 0, 1)
        p_y_x_ote_softmax_gat = p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)

        stance_cat = torch.cat((stance_cls_softmax.contiguous().view(-1, 1, 3), stance_gat_softmax), 1)
        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance),
                                    self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim



        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        # a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1)
        # ts_cat1=torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1).contiguous().view(-1, 2, 300)
        # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(ts_cat1, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
        # alpha_ts1 = F.softmax(alpha_ts1.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde = torch.matmul(alpha_ts1, ts_cat1)
        # p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, 100, 3)
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        # p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        # p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        # sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        # stance_gat = self.stance_gat(sequence_gat_output_stance)
        # stance_gat_softmax = F.softmax(stance_gat, dim=2)
        # sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        # a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        # b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        # c=torch.cat((a,b),1)
        # stance_cat = torch.cat((c, stance_gat_softmax), 1)
        # alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        # alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        # stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        # p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        # stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        #                                                                    3).contiguous().view(-1, 1, 3)
        #
        # transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
        #     -1, 1, 3)
        # aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
        #                                                                                                     self.max_len,
        #                                                                                                     3)
        # # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)
        #
        # #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # ts_cat2_a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1), ote2ts2.unsqueeze(1)), 1)
        # ts_cat2 = ts_cat2_a.contiguous().view(-1, 3, 300)
        # alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(ts_cat2, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(1, 2)
        # alpha_ts2 = F.softmax(alpha_ts2.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, ts_cat2)
        # p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, 100, 3)


        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
                #return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 , (
                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1222_2(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1222_2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.W_stance_gate = nn.Parameter(torch.FloatTensor(768, 300))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(300))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))
        # self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        # nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        # self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate.data, gain=1)
        # self.W_ts_gate2 = nn.Parameter(torch.FloatTensor(100, 3, 1))
        # nn.init.xavier_uniform_(self.W_ts_gate2.data, gain=1)
        # self.bias = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias.data, gain=1)
        # self.bias2 = nn.Parameter(torch.FloatTensor(100,3))
        # nn.init.xavier_uniform_(self.bias2.data, gain=1)

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]



        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2, 0, 1)
        p_y_x_ote_softmax_gat = p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        #stance_gat = self.stance_gat(sequence_gat_output_stance)
        #stance_gat_softmax = F.softmax(stance_gat, dim=2)

        stance_cls_output = outputs[1]
        #stance = self.stance(stance_cls_output)
        #stance_cls_softmax = F.softmax(stance, dim=1)

        stance_cat = torch.cat((stance_cls_output.unsqueeze(1), sequence_gat_output_stance), 1)
        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance),
                                    self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log_ = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 768)  # batch_size x 2*hidden_dim

        stance_log = self.stance(stance_log_)

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #p_y_x_ts_tilde = ote2ts.unsqueeze(2) + p_y_x_ts_softmax.unsqueeze(2)#普通加和
        #加权
        # a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1)
        # ts_cat1=torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1)), 1).contiguous().view(-1, 2, 300)
        # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(ts_cat1, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
        # alpha_ts1 = F.softmax(alpha_ts1.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde = torch.matmul(alpha_ts1, ts_cat1)
        # p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, 100, 3)
        p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        # p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        # p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        # sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        # stance_gat = self.stance_gat(sequence_gat_output_stance)
        # stance_gat_softmax = F.softmax(stance_gat, dim=2)
        # sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        # a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        # b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        # c=torch.cat((a,b),1)
        # stance_cat = torch.cat((c, stance_gat_softmax), 1)
        # alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        # alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        # stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        # p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        # stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        #                                                                    3).contiguous().view(-1, 1, 3)
        #
        # transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
        #     -1, 1, 3)
        # aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
        #                                                                                                     self.max_len,
        #                                                                                                     3)
        # # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)
        #
        # #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        # ts_cat2_a = torch.cat((ote2ts.unsqueeze(1), p_y_x_ts_softmax.unsqueeze(1), ote2ts2.unsqueeze(1)), 1)
        # ts_cat2 = ts_cat2_a.contiguous().view(-1, 3, 300)
        # alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(ts_cat2, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(1, 2)
        # alpha_ts2 = F.softmax(alpha_ts2.sum(1, keepdim=True), dim=2)
        # p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, ts_cat2)
        # p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, 100, 3)


        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                # stance_loss3 = loss_fct(
                #     stance_cls_softmax.view(-1, 3),
                #     stance_label.view(-1)
                # )

                # stance_loss4 = loss_fct(
                #     stance_gat_softmax.view(-1, 3),
                #     stance_label.view(-1)
                # )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1 + loss_ts2 + loss_ts3 + stance_loss + stance_loss2 + argument_t_loss, (
                return loss_ote + loss_ts1 + stance_loss + stance_loss2 + argument_t_loss, (
                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts.decode(ote2ts, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1223_1(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1223_1, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, 100))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(100))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(100, 3, 100))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(100, 100, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(100, 2, 100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, 100))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(100, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        ts_cat1=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2).contiguous().view(-1, 200, 3)
        b=torch.matmul(a, self.W_ts1)
        bb=b + self.bias_ts1
        bbbb = torch.matmul(bb, self.V_ts1)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, 100, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1 + loss_ts2 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1223_2(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1223_2, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
        alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
                #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

# tensor(0.8157, device='cuda:0', grad_fn=<NegBackward0>)
# tensor(1.0232, device='cuda:0', grad_fn=<NegBackward0>)
# tensor(0.9986, device='cuda:0', grad_fn=<NegBackward0>)
# tensor(1.1047, device='cuda:0', grad_fn=<NllLossBackward0>)
# tensor(1.3393, device='cuda:0', grad_fn=<NllLossBackward0>)
# tensor(0.6985, device='cuda:0', grad_fn=<NllLossBackward0>)
# tensor(0.6739, device='cuda:0', grad_fn=<NllLossBackward0>)
# tensor(0.7818, device='cuda:0', grad_fn=<NllLossBackward0>)


class Token_indicator_multitask_se_1227(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1227, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
        alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_1229(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1229, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 100
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = ote_tag_vocab[t]
            for nt in next_tags:
                ts_id = ts_tag_vocab[nt]
                self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        print(self.transition_scores)
        self.dim=100
        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.dim))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(300, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.dim))
        # self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        # nn.init.xavier_uniform_(self.W_gate2.data, gain=1)
        # self.W_ts_gate = nn.Parameter(torch.FloatTensor(100, 2, 1))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        self.sequence_stance = torch.nn.Linear(300, 3)
        self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,
                start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界

        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)

        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1, self.max_len, 3)

        a = torch.cat((ote2ts.unsqueeze(1).contiguous().view(-1, 1, 300), p_y_x_ts_softmax.unsqueeze(1).contiguous().view(-1, 1, 300)), 1)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
        alpha_ts1 = F.softmax(alpha_ts1.sum(1, keepdim=True), dim=2)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###
        # 加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        # p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)

        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2, 0, 1)
        p_y_x_ote_softmax_gat = p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        # stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        # stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        # stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a = stance_cls_softmax.contiguous().view(-1, 1, 3)
        b = sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c = torch.cat((a, b), 1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance),
                                    self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim

        # 第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        # stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                              self.max_len,
                                                                                                              3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        # sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(1).contiguous().view(-1, 1, 300), p_y_x_ts_softmax.unsqueeze(1).contiguous().view(-1, 1, 300)), 1)
        aa = torch.cat((a, ote2ts2.unsqueeze(1).contiguous().view(-1, 1, 300)), 1)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(1, 2)
        alpha_ts2 = F.softmax(alpha_ts2.sum(1, keepdim=True), dim=2)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        # p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        # stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                # return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                # return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                    loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))


class Token_indicator_multitask_se_1231(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_1231, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.crf_ts2 = CRF(3, batch_first=self.batch_first)
        self.crf_ts3 = CRF(3, batch_first=self.batch_first)
        self.crf_ts4 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
        alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts5 = -self.crf_ts4(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts4 = -self.crf_ts3(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts2(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts1(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5 + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
                return loss_ote + loss_ts1 + stance_loss  + argument_t_loss, (
                #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, loss_ts3 ,loss_ts4 ,loss_ts5, stance_loss, stance_loss2, argument_t_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts1.decode(ote2ts, attention_mask.byte()),
                        self.crf_ts2.decode(ote2ts2, attention_mask.byte()),
                        self.crf_ts3.decode(p_y_x_ts_tilde, attention_mask.byte()),
                        self.crf_ts4.decode(p_y_x_ts_softmax, attention_mask.byte()),
                        torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_cross_0106(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_cross_0106, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # outputs = self.bert(
        #     argument_input_ids,
        #     attention_mask=argument_attention_mask,
        #     token_type_ids=argument_token_type_ids
        # )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
        alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = loss_fct(
                    stance_log.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss2 = loss_fct(
                    sequence_stance_output.view(-1, 3),
                    stance_label.view(-1)
                )
                #
                stance_loss3 = loss_fct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label.view(-1)
                )

                stance_loss4 = loss_fct(
                    stance_gat_softmax.view(-1, 3),
                    stance_label.view(-1)
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
                #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(stance_log, dim=1),
                        torch.argmax(sequence_stance_output, dim=1))

class Token_indicator_multitask_se_0111(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_se_0111, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.pro_bert = BertModel.from_pretrained(model_name)
        self.con_bert = BertModel.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        self.pro_gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)

        self.pro_gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)

        self.con_gru_ote = nn.GRU(input_size=768,
                                  hidden_size=self.dim_ote_h,
                                  num_layers=1,
                                  bias=True,
                                  dropout=0,
                                  batch_first=True,
                                  bidirectional=True)

        self.con_gru_ts = nn.GRU(input_size=2 * self.dim_ote_h,
                                 hidden_size=self.dim_ote_h,
                                 num_layers=1,
                                 bias=True,
                                 dropout=0,
                                 batch_first=True,
                                 bidirectional=True)

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
        # transition_path = {'I': ['PRO', 'CON'], 'O': ['O']}
        # self.transition_scores = torch.zeros((2, 3))
        #
        # ote_tag_vocab = {'O': 0, 'I': 1}
        # ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
        # for t in transition_path:
        #     next_tags = transition_path[t]
        #     n_next_tag = len(next_tags)
        #     ote_id = ote_tag_vocab[t]
        #     for nt in next_tags:
        #         ts_id = ts_tag_vocab[nt]
        #         self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        # print(self.transition_scores)

        self.pro_stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.con_stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.pro_stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.con_stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.pro_stm_lm_activation = nn.Tanh()
        self.con_stm_lm_activation = nn.Tanh()
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.pro_W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.pro_W_gate.data, gain=1)
        self.con_W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.con_W_gate.data, gain=1)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ote = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ote.data, gain=1)
        self.V_ote = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ote.data, gain=1)
        self.bias_ote = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.pro_stance = torch.nn.Linear(768, 2)
        self.con_stance = torch.nn.Linear(768, 2)
        self.pro_stance_gat = torch.nn.Linear(768, 2)
        self.con_stance_gat = torch.nn.Linear(768, 2)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_pro_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_con_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_pro_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.fc_con_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)
        self.pro_crf_ote = CRF(2, batch_first=self.batch_first)
        self.con_crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.pro_crf_ts = CRF(3, batch_first=self.batch_first)
        self.con_crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.stance_pro_mask = torch.tensor(np.array([[1., 0 ,0], [0, 1., 0]]), dtype=torch.float32)
        self.stance_con_mask = torch.tensor(np.array([[1., 0 ,0], [0, 0, 1.]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask, argument_token_type_ids,
                labels=None, pro_labels=None, con_labels=None, IO_labels=None, pro_IO_labels=None, con_IO_labels=None, stance_label=None, pro_stance_label=None, con_stance_label=None, argument_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        pro_outputs = self.pro_bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        pro_sequence_output = pro_outputs[0]
        pro_stance_cls_output = pro_outputs[1]
        pro_stance = self.pro_stance(pro_stance_cls_output)
        pro_stance_cls_softmax = F.softmax(pro_stance, dim=1)#pro的stance

        pro_sequence_output = self.dropout(pro_sequence_output)
        pro_ote_hs, _ = self.pro_gru_ote(pro_sequence_output.view(pro_sequence_output.shape[0], -1, 768))
        pro_stm_lm_hs = self.pro_stm_lm(pro_ote_hs)
        pro_stm_lm_hs = self.dropout(pro_stm_lm_hs)  # 只判断边界



        #
        pro_ts_hs, _ = self.pro_gru_ts(pro_ote_hs)
        # 边界的基础上判断情感

        pro_ts_hs_transpose = torch.transpose(pro_ts_hs, 0, 1)
        pro_ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(pro_ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = pro_ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                pro_ts_ht = pro_ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(pro_ts_ht, self.pro_W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * pro_ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                pro_ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                pro_ts_hs_tilde = torch.cat((pro_ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        pro_ts_hs = torch.transpose(pro_ts_hs_tilde, 0, 1)
        pro_stm_lm_ts = self.pro_stm_ts(pro_ts_hs)
        pro_stm_lm_ts = self.dropout(pro_stm_lm_ts)


        pro_p_y_x_ote = self.fc_pro_ote(pro_stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(pro_p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_pro_ts(pro_stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        pro_stance_cls_softmax2 = torch.matmul(pro_stance_cls_softmax,self.stance_pro_mask.to('cuda'))
        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = pro_stance_cls_softmax2.unsqueeze(1).expand(pro_sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)#立场的扩展

        transs = self.trans.to('cuda').unsqueeze(1).expand(pro_sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        pro_ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)





        # 再加一个交叉熵验证
        con_outputs = self.con_bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        con_sequence_output = con_outputs[0]
        con_stance_cls_output = con_outputs[1]
        con_stance = self.con_stance(con_stance_cls_output)
        con_stance_cls_softmax = F.softmax(con_stance, dim=1)



        con_sequence_output = self.dropout(con_sequence_output)
        con_ote_hs, _ = self.con_gru_ote(con_sequence_output.view(con_sequence_output.shape[0], -1, 768))
        con_stm_lm_hs = self.con_stm_lm(con_ote_hs)
        con_stm_lm_hs = self.dropout(con_stm_lm_hs)  # 只判断边界



        #
        con_ts_hs, _ = self.con_gru_ts(con_ote_hs)
        # 边界的基础上判断情感

        con_ts_hs_transpose = torch.transpose(con_ts_hs, 0, 1)
        con_ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(con_ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = con_ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                con_ts_ht = con_ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(con_ts_ht, self.con_W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * con_ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                con_ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                con_ts_hs_tilde = torch.cat((con_ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        con_ts_hs = torch.transpose(con_ts_hs_tilde, 0, 1)
        con_stm_lm_ts = self.con_stm_ts(con_ts_hs)
        con_stm_lm_ts = self.dropout(con_stm_lm_ts)


        con_p_y_x_ote = self.fc_con_ote(con_stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(con_p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_con_ts(con_stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
        con_stance_cls_softmax2 = torch.matmul(con_stance_cls_softmax,self.stance_con_mask.to('cuda'))

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = con_stance_cls_softmax2.unsqueeze(1).expand(con_sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)#立场的扩展

        transs = self.trans.to('cuda').unsqueeze(1).expand(con_sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        con_ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        #
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)
        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13


        #ote2ts = pro_ote2ts + con_ote2ts

        #ote2ts = pro_ote2ts + con_ote2ts
        ote = torch.cat((pro_ote2ts.unsqueeze(2), con_ote2ts.unsqueeze(2)), 2)
        #aa=torch.matmul(ote, self.W_ote)
        #b=torch.tanh(torch.matmul(ote, self.W_ote) + self.bias_ote)
        #c=torch.matmul(torch.tanh(torch.matmul(ote, self.W_ote) + self.bias_ote), self.V_ote)
        alpha_ote = torch.matmul(torch.tanh(torch.matmul(ote, self.W_ote) + self.bias_ote), self.V_ote).transpose(2, 3)
        alpha_ote = F.softmax(alpha_ote.sum(2, keepdim=True), dim=3)
        ote2ts = torch.matmul(alpha_ote, ote)
        ote2ts = ote2ts.contiguous().view(-1, self.max_len, 3)
        #使用相似度计算，相似度高的话，都为空的话均，相似度低的话是一个带标签，一个不带

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###




        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()

                pro_stance_loss = loss_fct(
                    pro_stance_cls_softmax.view(-1, 2),
                    pro_stance_label.view(-1)
                )
                con_stance_loss = loss_fct(
                    con_stance_cls_softmax.view(-1, 2),
                    con_stance_label.view(-1)
                )
                #


                # ote2ts
                loss_ote_pro = -self.pro_crf_ote(pro_p_y_x_ote, pro_IO_labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')
                loss_ote_con = -self.con_crf_ote(con_p_y_x_ote, con_IO_labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')
                #loss_ote = -self.crf_ote(con_p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts_pro = -self.pro_crf_ts(con_ote2ts, con_labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')
                loss_ts_con = -self.con_crf_ts(pro_ote2ts, pro_labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')
                loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels.view(-1, self.max_len), attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote_pro + loss_ote_con + loss_ote + loss_ts + loss_ts_pro + loss_ts_con + pro_stance_loss + con_stance_loss, (
                #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote_pro, loss_ote_con, loss_ote, loss_ts, loss_ts_pro, loss_ts_con, pro_stance_loss, con_stance_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.pro_crf_ote.decode(pro_p_y_x_ote, attention_mask.byte()),
                        self.con_crf_ote.decode(con_p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.argmax(pro_stance_cls_softmax, dim=1),
                        torch.argmax(con_stance_cls_softmax, dim=1))


class Token_indicator_multitask_new_0117(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_indicator_multitask_new_0117, self).__init__()
        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(768, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)

    def forward(self, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None, start_label=None, end_label=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]

        stance_cls_output = outputs[1]
        stance = self.stance(stance_cls_output)
        stance_cls_softmax = F.sigmoid(stance)
        #stance_cls_softmax = F.softmax(stance, dim=1)

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界


        start_p = self.drnn_start_linear(ote_hs)
        end_p = self.drnn_end_linear(ote_hs)

        #
        ts_hs, _ = self.gru_ts(ote_hs)
        # 边界的基础上判断情感

        ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
        ts_hs_tilde = []
        h_tilde_tm1 = object
        for i in range(ts_hs_transpose.shape[0]):
            if i == 0:
                h_tilde_t = ts_hs_transpose[i]
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)


        p_y_x_ote = self.fc_ote(stm_lm_hs)
        p_y_x_ote_softmax = F.sigmoid(p_y_x_ote)
        #p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
        p_y_x_ts = self.fc_ts(stm_lm_ts)
        p_y_x_ts_softmax = F.sigmoid(p_y_x_ts)  # 13
        #p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

        p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                       3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)

        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
        alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
        p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###
        #加权
        # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
        # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
        #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
        sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)


        # 再加一个交叉熵验证

        p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
        p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
        sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
        stance_gat = self.stance_gat(sequence_gat_output_stance)
        stance_gat_softmax = F.sigmoid(stance_gat)
        #stance_gat_softmax = F.softmax(stance_gat, dim=2)
        sequence_stance_output_softmax = F.sigmoid(sequence_stance_output)
        #sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)

        #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
        #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
        #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax

        a=stance_cls_softmax.contiguous().view(-1, 1, 3)
        b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
        c=torch.cat((a,b),1)
        stance_cat = torch.cat((c, stance_gat_softmax), 1)

        alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
        alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
        stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim


        #第二层
        p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
        #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
        stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
                                                                           3).contiguous().view(-1, 1, 3)

        transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            -1, 1, 3)
        aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
        ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
                                                                                                            self.max_len,
                                                                                                            3)
        # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
        # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
        # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)

        #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
        a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
        aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
        alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
        alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
        p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
        p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
        #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax

        #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)

        argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率

        if self.use_crf:
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                mt_loss_ct = nn.MultiLabelSoftMarginLoss()
                start_loss = loss_fct(
                    start_p.view(-1, 2),
                    start_label.view(-1)
                )
                end_loss = loss_fct(
                    end_p.view(-1, 2),
                    end_label.view(-1)
                )
                stance_loss = mt_loss_ct(
                    stance_log.view(-1, 3),
                    stance_label
                )
                #
                stance_loss2 = mt_loss_ct(
                    sequence_stance_output.view(-1, 3),
                    stance_label
                )
                #
                stance_loss3 = mt_loss_ct(
                    stance_cls_softmax.view(-1, 3),
                    stance_label
                )

                stance_loss4 = mt_loss_ct(
                    stance_gat.view(-1, 3),
                    stance_label
                )
                argument_t_loss = loss_fct(
                    argument_t.view(-1, 2),
                    argument_label.view(-1)
                )

                # ote2ts
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')

                #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
                #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
                #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
                loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
                # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            else:  # inference
                return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
                        self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.sigmoid(stance_log),
                        torch.sigmoid(sequence_stance_output))


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)  # 使用边长的LSTM，必须要从大到小进行排序，
        x_unsort_idx = torch.argsort(x_sort_idx).long()  # 再返回正常顺序。
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)

class Token_pipeline_0211(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0211, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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
        ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
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
        self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        self.attention = self_attention_layer(768)
        self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None,trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        if md == 'IO_inference':
            stance_cls_output = outputs[1]
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts3 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            # p_y_x_ote_softmax = F.sigmoid(p_y_x_ote)
            # #p_y_x_ote_softmax = F.softmax(p_y_x_ote, dim=-1)
            # p_y_x_ts = self.fc_ts(stm_lm_ts)
            # p_y_x_ts_softmax = F.sigmoid(p_y_x_ts)  # 13
            # #p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            #
            # p_y_x_ote_softmax_unsequence = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
            # stance_softmax_unsequence = stance_cls_softmax.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
            #                                                                3).contiguous().view(-1, 1, 3)
            #
            # transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            #     -1, 1, 3)
            # aaaa = torch.cat((transs, stance_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
            # ote2ts = torch.matmul(p_y_x_ote_softmax_unsequence, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
            #                                                                                                     self.max_len,
            #                                                                                                     3)
            #
            # a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            # p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)###

            #
            # #加权
            # # a=torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # # #p_y_x_ts_tilde = torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias
            # # p_y_x_ts_tilde = torch.tanh(torch.matmul(a.permute(0, 1, 3, 2), self.W_ts_gate.cuda()).contiguous().view(-1, self.max_len, 3)+ self.bias)
            # #p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            # sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
            #
            #
            # # 再加一个交叉熵验证
            #
            # p_y_x_ote_softmax_T = p_y_x_ote_softmax.permute(2,0,1)
            # p_y_x_ote_softmax_gat=p_y_x_ote_softmax_T[1].unsqueeze(1)
            # sequence_gat_output_stance = torch.bmm(p_y_x_ote_softmax_gat, sequence_output)
            # stance_gat = self.stance_gat(sequence_gat_output_stance)
            # stance_gat_softmax = F.sigmoid(stance_gat)
            # #stance_gat_softmax = F.softmax(stance_gat, dim=2)
            # sequence_stance_output_softmax = F.sigmoid(sequence_stance_output)
            # #sequence_stance_output_softmax = F.softmax(sequence_stance_output, dim=1)
            #
            # #stance_cat= torch.cat((torch.cat((stance_cls_softmax.unsqueeze(1),stance_gat_softmax),1),sequence_stance_output_softmax.unsqueeze(1)),1)
            # #stance_log = torch.matmul(stance_cat, self.W_stance_gate.cuda()).contiguous().view(-1, 3)
            # #stance_log = stance_cls_softmax + stance_gat_softmax.contiguous().view(-1, 3) + sequence_stance_output_softmax
            #
            # a=stance_cls_softmax.contiguous().view(-1, 1, 3)
            # b=sequence_stance_output_softmax.contiguous().view(-1, 1, 3)
            # c=torch.cat((a,b),1)
            # stance_cat = torch.cat((c, stance_gat_softmax), 1)
            #
            # alpha_stance = torch.matmul(torch.tanh(torch.matmul(stance_cat, self.W_stance_gate) + self.bias_stance), self.V_stance).transpose(1, 2)
            # alpha_stance = F.softmax(alpha_stance.sum(1, keepdim=True), dim=2)
            # stance_log = torch.matmul(alpha_stance, stance_cat).contiguous().view(-1, 3)  # batch_size x 2*hidden_dim
            #
            #
            # #第二层
            # p_y_x_ote_softmax_unsequence2 = p_y_x_ote_softmax.unsqueeze(3).contiguous().view(-1, 1, 2)
            # #stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
            # stance_log_softmax_unsequence = stance_log.unsqueeze(1).expand(sequence_output.shape[0], self.max_len,
            #                                                                    3).contiguous().view(-1, 1, 3)
            #
            # transs = self.trans.to('cuda').unsqueeze(1).expand(sequence_output.shape[0], self.max_len, 3).contiguous().view(
            #     -1, 1, 3)
            # aaaa = torch.cat((transs, stance_log_softmax_unsequence), 1).contiguous().view(-1, 2, 3)
            # ote2ts2 = torch.matmul(p_y_x_ote_softmax_unsequence2, aaaa * self.mask2.to('cuda')).contiguous().view(-1,
            #                                                                                                     self.max_len,
            #                                                                                                     3)
            # # a3 = torch.cat([ote2ts.unsqueeze(2), ote2ts2.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)], 2)
            # # #p_y_x_ts_tilde2 = torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2
            # # p_y_x_ts_tilde2 = torch.tanh(torch.matmul(a3.permute(0, 1, 3, 2), self.W_ts_gate2.cuda()).contiguous().view(-1, self.max_len, 3) + self.bias2)
            #
            # #sequence_stance_output = self.attention2(p_y_x_ts_tilde, attention_mask)
            # a = torch.cat((ote2ts.unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # aa = torch.cat((a, ote2ts2.unsqueeze(2)), 2)
            # alpha_ts2 = torch.matmul(torch.tanh(torch.matmul(aa, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts2 = F.softmax(alpha_ts2.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde2 = torch.matmul(alpha_ts2, aa)
            # p_y_x_ts_tilde2 = p_y_x_ts_tilde2.contiguous().view(-1, self.max_len, 3)  ###
            # #p_y_x_ts_tilde2 = ote2ts + ote2ts2 + p_y_x_ts_softmax
            #
            # #stance_log = F.softmax(stance, dim=1) + F.softmax(sequence_stance_output, dim=1)
            #
            # argument_t = torch.matmul(stance_log, self.stance2argument.to('cuda'))  # stance到argument的转移概率
            #
            # if self.use_crf:
            #     if labels is not None:
            #         loss_fct = nn.CrossEntropyLoss()
            #         mt_loss_ct = nn.MultiLabelSoftMarginLoss()
            #         start_loss = loss_fct(
            #             start_p.view(-1, 2),
            #             start_label.view(-1)
            #         )
            #         end_loss = loss_fct(
            #             end_p.view(-1, 2),
            #             end_label.view(-1)
            #         )
            #         stance_loss = mt_loss_ct(
            #             stance_log.view(-1, 3),
            #             stance_label
            #         )
            #         #
            #         stance_loss2 = mt_loss_ct(
            #             sequence_stance_output.view(-1, 3),
            #             stance_label
            #         )
            #         #
            #         stance_loss3 = mt_loss_ct(
            #             stance_cls_softmax.view(-1, 3),
            #             stance_label
            #         )
            #
            #         stance_loss4 = mt_loss_ct(
            #             stance_gat.view(-1, 3),
            #             stance_label
            #         )
            #         argument_t_loss = loss_fct(
            #             argument_t.view(-1, 2),
            #             argument_label.view(-1)
            #         )
            #
            #         # ote2ts
            #         loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
            #         loss_ts5 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')
            #         loss_ts4 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
            #         loss_ts3 = -self.crf_ts(ote2ts2, labels, attention_mask.byte(), reduction='token_mean')
            #         loss_ts2 = -self.crf_ts(ote2ts, labels, attention_mask.byte(), reduction='token_mean')
            #         loss_ts1 = -self.crf_ts(p_y_x_ts_tilde2, labels, attention_mask.byte(), reduction='token_mean')
            #
            #         #return loss_ote + loss_ts1 + stance_loss + argument_t_loss + start_loss + end_loss, (
            #         #return loss_ote + loss_ts1  + loss_ts2  +loss_ts3  +loss_ts4  + loss_ts5  +stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
            #         return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss, (
            #         #return loss_ote + loss_ts1  + stance_loss + stance_loss2 + stance_loss3 + stance_loss4 + argument_t_loss + start_loss + end_loss, (
            #         loss_ote, loss_ts1, loss_ts2, stance_loss, stance_loss2, argument_t_loss, start_loss, end_loss)
            #         # return loss_ote + loss_ts1 + loss_ts2 + stance_loss1 + stance_loss2 + argument_loss + argument_t_loss, (loss_ote, loss_ts1,  stance_loss1, argument_loss, argument_t_loss, loss_ts2, stance_loss2)
            #     else:  # inference
            #         return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()),
            #                 self.crf_ts.decode(p_y_x_ts_tilde2, attention_mask.byte()),
            #                 self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()), torch.sigmoid(stance_log),
            #                 torch.sigmoid(sequence_stance_output))




class Token_pipeline_0228(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0228, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        # transition_path = {'I': ['PRO', 'CON'], 'O': ['O']}
        # self.transition_scores = torch.zeros((2, 3))
        #
        # ote_tag_vocab = {'O': 0, 'I': 1}
        # ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
        # for t in transition_path:
        #     next_tags = transition_path[t]
        #     n_next_tag = len(next_tags)
        #     ote_id = ote_tag_vocab[t]
        #     for nt in next_tags:
        #         ts_id = ts_tag_vocab[nt]
        #         self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        # print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            # ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            # ts_hs_tilde = []
            # h_tilde_tm1 = object
            # for i in range(ts_hs_transpose.shape[0]):
            #     if i == 0:
            #         h_tilde_t = ts_hs_transpose[i]
            #         # ts_hs_tilde = h_tilde_t.view(1, -1)
            #     else:
            #         # t-th hidden state for the task targeted sentiment
            #         ts_ht = ts_hs_transpose[i]
            #         gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
            #         # a= (1 - gt)
            #         h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
            #         # print(h_tilde_t)
            #     # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            #     if i == 0:
            #         ts_hs_tilde = h_tilde_t.unsqueeze(0)
            #     else:
            #         ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            #     # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)
            #
            #     h_tilde_tm1 = h_tilde_t
            # ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())
class Token_pipeline_0301(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0301, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0301_dot(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0301_dot, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # alpha = torch.matmul(ote_stance, topic_.transpose(1, 0))
            alpha = torch.matmul(topic_, ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0301_general(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0301_general, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # alpha = torch.matmul(ote_stance, topic_.transpose(1, 0))
            alpha = torch.matmul(torch.matmul(topic_, self.W_ts1), ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0301_perceptron(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0301_perceptron, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts11 = nn.Parameter(torch.FloatTensor(300, 300))
        self.W_ts22 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts11.data, gain=1)
        nn.init.xavier_uniform_(self.W_ts22.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # a=torch.matmul(topic_, self.W_ts11)
            # b =torch.matmul(ote_stance, self.W_ts22)
            alpha = torch.matmul(F.tanh(torch.matmul(topic_, self.W_ts11) + torch.matmul(ote_stance, self.W_ts22)),
                                 self.V_ts1).transpose(1, 2)
            # alpha2 = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_0302(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_0302, self).__init__()
        self.num_labels = 7
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

        transition_path = {'I': ['PRO', 'CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'CON': 1, 'PRO': 2}
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
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.dropout = nn.Dropout(0.2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, IO_labels=None):
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
                # ts_hs_tilde = h_tilde_t.view(1, -1)
            else:
                # t-th hidden state for the task targeted sentiment
                ts_ht = ts_hs_transpose[i]
                gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                # a= (1 - gt)
                h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                # print(h_tilde_t)
            # ts_hs_tilde.append(h_tilde_t.view(1,-1))
            if i == 0:
                ts_hs_tilde = h_tilde_t.unsqueeze(0)
            else:
                ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
            # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

            h_tilde_tm1 = h_tilde_t
        ts_hs = torch.transpose(ts_hs_tilde, 0, 1)

        stm_lm_ts = self.stm_ts(ts_hs)
        stm_lm_ts = self.dropout(stm_lm_ts)
        #     if t==0:
        #         ts_hs_tildes=ts_hs_tilde.unsqueeze(0)
        #     else:
        p_y_x_ote = self.fc_ote(stm_lm_hs)

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
        p_y_x_ts_tilde = torch.mul(alpha.expand([sequence_output.shape[0], 90, 3]), ote2ts) + torch.mul((1 - alpha),
                                                                                                         p_y_x_ts_softmax)
        if labels is not None:  # training
            loss_fct = nn.CrossEntropyLoss()
            loss_ote = loss_fct(
                p_y_x_ote.view(-1, self.dim_ote_y),
                IO_labels.view(-1)
            )
            loss_ts = loss_fct(
                p_y_x_ts_tilde.view(-1, self.dim_ts_y),
                labels.view(-1)
            )
            return loss_ts+loss_ote
        else:  # inference
            #return (torch.argmax(p_y_x_ote, dim=2),torch.argmax(p_y_x_ts_tilde, dim=2))
            return torch.argmax(p_y_x_ts_tilde, dim=2)
        # if labels is not None:
        #     loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
        #     loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
        #     return loss_ts + loss_ote
        # else:  # inference
        #     return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()))

class Token_0303(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_0303, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 60
        self.dim_ote_y = 3
        self.dim_ts_y = 3
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)

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

        transition_path = {'I': ['PRO', 'CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'CON': 1, 'PRO': 2}
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
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.dropout = nn.Dropout(0.4)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(3, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, IO_labels=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        stm_lm_hs = self.stm_lm(ote_hs)
        #stm_lm_hs = self.dropout(stm_lm_hs)#只判断边界
        p_y_x_ote = self.fc_ote(stm_lm_hs)


        if labels is not None:  # training
            loss_fct = nn.CrossEntropyLoss()
            loss_ts = loss_fct(
                p_y_x_ote.view(-1, self.dim_ts_y),
                labels.view(-1)
            )
            return loss_ts
        else:  # inference
            return (torch.argmax(p_y_x_ote, dim=2))
        # if labels is not None:
        #     loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
        #     loss_ts = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
        #     return loss_ts + loss_ote
        # else:  # inference
        #     return (self.crf_ote.decode(p_y_x_ote, attention_mask.byte()), self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte()))

class Token_0303_LSTM_CRF(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_0303_LSTM_CRF, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 3
        self.dim_ts_y = 3
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)

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

        transition_path = {'I': ['PRO', 'CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'CON': 1, 'PRO': 2}
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
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.dropout = nn.Dropout(0.1)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        #if self.use_crf:
        self.crf_ote = CRF(3, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, IO_labels=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #stm_lm_hs = self.stm_lm(ote_hs)
        #stm_lm_hs = self.dropout(ote_hs)#只判断边界
        p_y_x_ote = self.fc_ote(ote_hs)


        # if labels is not None:  # training
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss_ts = loss_fct(
        #         p_y_x_ote.view(-1, self.dim_ts_y),
        #         labels.view(-1)
        #     )
        #     return loss_ts
        # else:  # inference
        #     return (torch.argmax(p_y_x_ote, dim=2))
        if labels is not None:
            loss_ts = -self.crf_ts(p_y_x_ote, labels, attention_mask.byte(), reduction='token_mean')
            return loss_ts
        else:  # inference
            return (self.crf_ts.decode(p_y_x_ote, attention_mask.byte()))

class Token_pipeline_0304(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0304, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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

        # transition_path = {'I': ['PRO', 'CON'], 'O': ['O']}
        # self.transition_scores = torch.zeros((2, 3))
        #
        # ote_tag_vocab = {'O': 0, 'I': 1}
        # ts_tag_vocab = {'O': 0, 'PRO': 2, 'CON': 1}
        # for t in transition_path:
        #     next_tags = transition_path[t]
        #     n_next_tag = len(next_tags)
        #     ote_id = ote_tag_vocab[t]
        #     for nt in next_tags:
        #         ts_id = ts_tag_vocab[nt]
        #         self.transition_scores[ote_id][ts_id] = 1.0 / n_next_tag
        # print(self.transition_scores)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts2 = -self.crf_ts(p_y_x_ts_softmax, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1 + loss_ts2
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())




class Token_0304_IDCNN_CRF(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_0304_IDCNN_CRF, self).__init__()
        self.num_labels = 7
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 3
        self.dim_ts_y = 3
        # self.tokenbert = BertForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=self.num_labels,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions
        # )
        self.bert = BertModel.from_pretrained(model_name)

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

        transition_path = {'I': ['PRO', 'CON'],
                           'O': ['O']}
        self.transition_scores = torch.zeros((2, 3))

        ote_tag_vocab = {'O': 0, 'I': 1}
        ts_tag_vocab = {'O': 0, 'CON': 1, 'PRO': 2}
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
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.dropout = nn.Dropout(0.4)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn = torch.nn.Linear(300, 3)
        #if self.use_crf:
        self.crf_ote = CRF(3, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, IO_labels=None):
        #outputs = self.tokenbert.bert(
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)



        sequence_output = sequence_output.permute(0, 2, 1)
        inner_last_output = F.relu(self.itdicnn0(sequence_output))
        # print(inner_last_output.shape)
        last_dims = 768
        last_output = inner_last_output
        # There could be output mismatch here, please consider concatenation if so
        block_unflat_scores = []
        for block in range(self.repeats):

            hidden_outputs = []
            total_output_width = 0
            inner_last_dims = last_dims
            inner_last_output = last_output
            block_input = last_output
            for layeri, conv in enumerate(self.itdicnn):
                h = F.relu(conv(inner_last_output))
                if self.take_layer[layeri]:
                    hidden_outputs.append(h)
                    total_output_width += 768
                inner_last_dims = 768
                if self.layer_residual:
                    inner_last_output = h + inner_last_output
                else:
                    inner_last_output = h

            h_concat = torch.cat(hidden_outputs, -1)
            if self.block_residual:
                last_output = self.dropout(h_concat) + block_input
            else:
                last_output = self.dropout(h_concat)
            last_dims = total_output_width

            block_unflat_scores.append(last_output)
        a=block_unflat_scores[0].permute(0,2,1)
        #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
        #stm_lm_hs = self.stm_lm(ote_hs)
        #stm_lm_hs = self.dropout(ote_hs)#只判断边界
        p_y_x_ote = self.id_cnn(block_unflat_scores[0].permute(0,2,1))


        # if labels is not None:  # training
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss_ts = loss_fct(
        #         p_y_x_ote.view(-1, self.dim_ts_y),
        #         labels.view(-1)
        #     )
        #     return loss_ts
        # else:  # inference
        #     return (torch.argmax(p_y_x_ote, dim=2))
        if labels is not None:
            loss_ts = -self.crf_ts(p_y_x_ote, labels, attention_mask.byte(), reduction='token_mean')
            return loss_ts
        else:  # inference
            return (self.crf_ts.decode(p_y_x_ote, attention_mask.byte()))

class Token_pipeline_0304_IDCNN_CRF(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0304_IDCNN_CRF, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))




            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())


class PointerNetwork_Multi(nn.Module):
    def __init__(self, answer_seq_len=2, is_GRU=True):
        super(PointerNetwork_Multi, self).__init__()
        self.encoder_hidden_size = 384
        self.decoder_hidden_size = 100
        self.answer_seq_len = answer_seq_len
        self.weight_size = 100
        self.is_GRU = is_GRU

        # self.Wd = nn.Linear(2 * self.encoder_hidden_size, 2 * self.encoder_hidden_size, bias=False) # blending encoder
        self.Wh = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias=False)  # blending encoder
        # self.Ws = nn.Linear(2 * self.encoder_hidden_size, self.decoder_hidden_size, bias=False) # blending encoder

        # self.emb = nn.Embedding(input_size, emb_size)  # embed inputs Embedding(11, 32)
        if is_GRU:
            # self.enc = nn.GRU(emb_size, hidden_size, batch_first=True)
            self.dec = nn.GRUCell(2 * self.encoder_hidden_size,
                                  self.decoder_hidden_size)  # GRUCell's input is always batch first
        else:
            # self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
            self.dec = nn.LSTMCell(2 * self.encoder_hidden_size,
                                   self.decoder_hidden_size)  # LSTMCell's input is always batch first

        self.W1 = nn.Linear(2 * self.encoder_hidden_size, self.weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(self.decoder_hidden_size, self.weight_size, bias=False)  # blending decoder

        self.vts = torch.nn.ModuleList([nn.Linear(self.weight_size, 1, bias=False) for i in range(self.answer_seq_len)])
        # self.vt = nn.Linear(self.weight_size, 1, bias=False) # scaling sum of enc and dec by v.T

    def forward(self, input, state_out, context_mask):
        """
        input:[batch, max_len, dim]
        context_mask:[batch, max_len, dim]
        """
        batch_size = input.size(0)
        # Encoding
        # encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = input.transpose(1, 0)  # (max_len, batch, dim)

        # Decoding states initialization
        decoder_input = state_out  # (bs, embd_size)
        hidden = self.Wh(state_out)  # (bs, h)
        cell_state = state_out

        max_len = input.size(1)
        probs_s = Variable(torch.zeros((self.answer_seq_len, batch_size, max_len))).long().cuda()
        probs_e = Variable(torch.zeros((self.answer_seq_len, batch_size, max_len))).long().cuda()

        probs = []
        # Decoding
        index_fun = 0
        for i in range(self.answer_seq_len * 2):  # range(M)
            if i != 0 and i % 2 == 0:
                index_fun += 1

            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden)  # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state))  # (bs, h), (bs, h)
            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)  # (max_len, bs, weigh_size)
            blend2 = self.W2(hidden)  # (batch, weih_size)
            blend_sum = F.tanh(blend1 + blend2)  # (max_len, bs, Wweig_size)

            func1 = self.vts[index_fun]
            out = func1(blend_sum).squeeze(-1)  # (max_len, bs)
            # print('out.size = ', out.size())
            # mask the output
            out = out.transpose(0, 1)  # [batch, max_len]
            out = out.masked_fill(context_mask == 0, -1e9)
            out = F.log_softmax(out, -1)  # [batch, max_len]
            # out = F.log_softmax(out.contiguous(), -1) # (bs, L)
            if i % 2 == 0:
                probs_s[index_fun] = out
            else:
                probs_e[index_fun] = out
            probs.append(out)

        probs = torch.stack(probs, dim=1)  # [ans * 2 * batch, max_len]
        # (bs, M, L)
        return probs, (probs_s.transpose(0, 1), probs_e.transpose(0,  1))  # [ans_lan*2*batch, max_len], [batch, ans_len, max_len], [batch, ans_len, max_len]


class Token_pipeline_0306_pointer(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0306_pointer, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        # self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        # self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
                                     hidden_size=self.dim_ote_h,
                                     num_layers=1,
                                     bias=True,
                                     dropout=0,
                                     batch_first=True,
                                     bidirectional=True)
        self.gru_ts = nn.GRU(input_size=768,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        # self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        # self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        # self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        # self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        # self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.1)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        # self.sequence_stance = torch.nn.Linear(300, 3)
        # self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8, 300).astype(np.float32)))

        self.pointer = PointerNetwork_Multi( answer_seq_len=2, is_GRU=True)
        #self.init_weights()

    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids,
                argument_attention_mask,
                argument_token_type_ids, span_label=None, labels=None, IO_labels=None, stance_label=None, argument_label=None,
                mask_positions=None, topic_indices=None, label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = outputs[0]
            # print('size = ', sequence_output.size())
            cls_out = sequence_output[:, 0, :]  # [batch, dim],第0个向量CLS
            sequence_output = self.dropout(sequence_output)
            # sequence_output = sequence_output.transpose(0,1) #[batch, max_len, dim]
            sequence_output = torch.mul(sequence_output, attention_mask.unsqueeze(-1).repeat(1, 1, sequence_output.size(
                -1)))  # [batch, max_len, dim] 将cls去掉

            probs, logits = self.pointer(sequence_output, cls_out, attention_mask)  # [batch, max_len], [batch, max_len]



            if span_label is not None:
                max_len = sequence_output.size(1)
                outputs = probs.view(-1, max_len)  # (bs*M, L)
                # print('y = ', y.size())
                span_label = span_label.reshape(-1)  # (bs*M)
                loss = F.nll_loss(outputs, span_label)
                return (loss, logits)
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #return loss_ote
            else:  # inference
                start_logits = logits[0]  # [batch, ans_len, max_len]
                end_logits = logits[1]  # [batch, ans_len, max_len]
                start_label = torch.argmax(start_logits, -1)  # [batch, ans_len]
                end_label = torch.argmax(end_logits, -1)  # [batch, ans_len]
                return (start_label, end_label)
            #return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a = torch.Tensor(mask_positions).view(90, -1).expand(90, 768).cuda()
            sequence_output_mask = sequence_output.view(90, 768) * a
            ote_stance, final = self.gru_ote_stance(
                sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(
                F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            #p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            # 边界的基础上判断情感

            stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2),
                                     self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_0311_bs(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0311_bs, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.2)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                return loss
            else:  # inference
                return torch.argmax(p_y_x_ote, dim=2)
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            forward=final[0].view(1,-1)
            backward=final[1].view(1,-1)
            stance = self.stance(torch.cat((forward,backward),1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(sequence_output)
            p_y_x_ts = self.fc_ts(sequence_output)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ote = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ts1 = loss_fct(
                    p_y_x_ts_tilde.view(-1, 3),
                    labels.view(-1)
                )
                # loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return torch.argmax(p_y_x_ts_tilde, dim=2)

class Token_pipeline_0312_layer1_crf(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_crf, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))

            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte())
                return loss
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            forward=final[0].view(1,-1)
            backward=final[1].view(1,-1)
            stance = self.stance(torch.cat((forward,backward),1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            # ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(ote)
            p_y_x_ts = self.fc_ts(sequence_output)

            # stm_lm_hs = self.stm_lm(ote_hs)
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            #ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_ts = self.stm_ts(ts_hs)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                #loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_lstm_crf(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_lstm_crf, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))

            ote = self.lm_ote(ote_hs)
            p_y_x_ote = self.fc_ote(ote_hs)

            if IO_labels is not None:
                loss = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte())
                return loss
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            forward=final[0].view(1,-1)
            backward=final[1].view(1,-1)
            stance = self.stance(torch.cat((forward,backward),1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ote = self.lm_ote(ote_hs)
            p_y_x_ote = self.fc_ote(ote)

            ts_hs, _ = self.gru_ts(ote_hs)
            ts = self.lm_ts(ts_hs)
            p_y_x_ts = self.fc_ts(ts)

            # stm_lm_hs = self.stm_lm(ote_hs)
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            #ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_ts = self.stm_ts(ts_hs)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote +  loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_idcnn_crf(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_idcnn_crf, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
                                                                                                                    3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_idcnn_crf_dot(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_idcnn_crf_dot, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None, topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))

            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # alpha = torch.matmul(ote_stance, topic_.transpose(1, 0))
            alpha = torch.matmul(topic_, ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
                                                                                                                    3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_idcnn_crf_general(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_idcnn_crf_general, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # alpha = torch.matmul(ote_stance, topic_.transpose(1, 0))
            alpha = torch.matmul(torch.matmul(topic_, self.W_ts1), ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
                                                                                                                    3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_idcnn_crf_perceptron(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_idcnn_crf_perceptron, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts11 = nn.Parameter(torch.FloatTensor(300, 300))
        self.W_ts22 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts11.data, gain=1)
        nn.init.xavier_uniform_(self.W_ts22.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            # ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            # a=torch.matmul(topic_, self.W_ts11)
            # b =torch.matmul(ote_stance, self.W_ts22)
            alpha = torch.matmul(F.tanh(torch.matmul(topic_, self.W_ts11) + torch.matmul(ote_stance, self.W_ts22)),
                                 self.V_ts1).transpose(1, 2)
            # alpha2 = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
                                                                                                                    3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_0313_layer2_attention(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0313_layer2_attention, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                return loss
            else:  # inference
                return torch.argmax(p_y_x_ote, dim=2)
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            # 长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(ote)
            p_y_x_ts = self.fc_ts(sequence_output)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ote = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                loss_ts = loss_fct(
                    p_y_x_ts.view(-1, 3),
                    labels.view(-1)
                )
                loss_ts1 = loss_fct(
                    p_y_x_ts_tilde.view(-1, 3),
                    labels.view(-1)
                )
                # loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return torch.argmax(p_y_x_ts_tilde, dim=2)

class Token_pipeline_0314_layer1_crf_b1c1(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_crf_b1c1, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))

            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte())
                return loss
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            # ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(ote)
            p_y_x_ts = self.fc_ts(sequence_output)

            # stm_lm_hs = self.stm_lm(ote_hs)
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            #ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_ts = self.stm_ts(ts_hs)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0314_layer1_lstm_crf_b1c1(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_lstm_crf_b1c1, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ######stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ##########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_0314_layer1_lstm_crf_b1c1_dot(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_lstm_crf_b1c1_dot, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ######stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])


            # alpha = torch.matmul(ote_stance, topic_.transpose(1, 0))
            alpha = torch.matmul(topic_, ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ##########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0314_layer1_lstm_crf_b1c1_general(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_lstm_crf_b1c1_general, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ######stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            # topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            # topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            alpha = torch.matmul(torch.matmul(topic_, self.W_ts1), ote_stance.transpose(1, 2))
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ##########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0314_layer1_lstm_crf_b1c1_perceptron(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_lstm_crf_b1c1_perceptron, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts11 = nn.Parameter(torch.FloatTensor(300, 300))
        self.W_ts22 = nn.Parameter(torch.FloatTensor(300, 300))
        nn.init.xavier_uniform_(self.W_ts11.data, gain=1)
        nn.init.xavier_uniform_(self.W_ts22.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ######stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])

            #a=torch.matmul(topic_, self.W_ts11)
            #b =torch.matmul(ote_stance, self.W_ts22)
            alpha = torch.matmul(F.tanh(torch.matmul(topic_, self.W_ts11) + torch.matmul(ote_stance, self.W_ts22)), self.V_ts1).transpose(1, 2)
            #alpha2 = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ##########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_0317_layer3_bs(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0317_layer3_bs, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                return loss
            else:  # inference
                return torch.argmax(p_y_x_ote, dim=2)
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            forward=final[0].view(1,-1)
            backward=final[1].view(1,-1)
            stance = self.stance(torch.cat((forward,backward),1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(sequence_output)
            p_y_x_ts = self.fc_ts(sequence_output)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13


            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = ote2ts+p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ote = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ts1 = loss_fct(
                    p_y_x_ts_tilde.view(-1, 3),
                    labels.view(-1)
                )
                # loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return torch.argmax(p_y_x_ts_tilde, dim=2)

class Token_pipeline_0317_layer1_crf_bc(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0317_layer1_crf_bc, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))

            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte())
                return loss
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #拼接final
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            # ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(ote)
            p_y_x_ts = self.fc_ts(sequence_output)

            # stm_lm_hs = self.stm_lm(ote_hs)
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            #ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_ts = self.stm_ts(ts_hs)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0317_layer1_lstm_crf_bc(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0317_layer1_lstm_crf_bc, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.2)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ############stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ###########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0317_layer1_idcnn_crf_bc(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0317_layer1_idcnn_crf_bc, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(200, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(200, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 200
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(200, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
            #                                                                                                         3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0318_layer3_b1c(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0318_layer3_b1c, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                return loss
            else:  # inference
                return torch.argmax(p_y_x_ote, dim=2)
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(sequence_output)
            p_y_x_ts = self.fc_ts(sequence_output)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13


            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = ote2ts+p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ote = loss_fct(
                    p_y_x_ote.view(-1, 2),
                    IO_labels.view(-1)
                )
                loss_ts = loss_fct(
                    p_y_x_ts.view(-1, 3),
                    labels.view(-1)
                )
                loss_ts1 = loss_fct(
                    p_y_x_ts_tilde.view(-1, 3),
                    labels.view(-1)
                )
                # loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return torch.argmax(p_y_x_ts_tilde, dim=2)

class Token_pipeline_0318_layer1_crf_b1c(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0318_layer1_crf_b1c, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.lm_ote = torch.nn.Linear(768, 2 * self.dim_ote_h)
        self.lm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(768, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(768, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))

            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            if IO_labels is not None:
                loss = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte())
                return loss
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            # ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #ote = self.lm_ote(sequence_output)
            p_y_x_ote = self.fc_ote(sequence_output)

            #ts = self.lm_ts(ote)
            p_y_x_ts = self.fc_ts(sequence_output)

            # stm_lm_hs = self.stm_lm(ote_hs)
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            #ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            #stm_lm_ts = self.stm_ts(ts_hs)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0318_layer1_lstm_crf_b1c(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0318_layer1_lstm_crf_b1c, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ##########stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ###########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(ote_hs)
            # 边界的基础上判断情感


            ###########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote +  loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0318_layer1_idcnn_crf_b1c(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0318_layer1_idcnn_crf_b1c, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            ##2
            sequence_output2 = block_unflat_scores[0]
            inner_last_output2 = F.relu(self.itdicnn2(sequence_output2))
            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            # a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            # alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
            #                                                                                                         3)
            # alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            # p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ote + loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_gru_crf_step(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_gru_crf_step, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.4)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, mask_positions=None, topic_indices=None, label_stance=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        #stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ##########stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

class Token_pipeline_idcnn_crf_step(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_idcnn_crf_step, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

class Token_pipeline_lstm_crf_step(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_lstm_crf_step, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.LSTM(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.LSTM(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ts = nn.LSTM(input_size=2 * self.dim_ote_h,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, mask_positions=None, topic_indices=None, label_stance=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        #stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ##########stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1),
                                 self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax

class Token_pipeline_0314_layer1_lstm_crf_b1c1_x(nn.Module):#注意力
    '''去掉2个dropout
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0314_layer1_lstm_crf_b1c1_x, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
        self.max_len = 90
        #self.topic_lstm = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True,only_use_last_hidden_state=True)
        #self.lstm_ote = DynamicLSTM(768, self.dim_ote_h, num_layers=1, batch_first=True, bidirectional=True)

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ts = nn.GRU(input_size=768,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100
        # self.W_stance_gate = nn.Parameter(torch.FloatTensor(3, self.dim))
        # nn.init.xavier_uniform_(self.W_stance_gate.data, gain=1)
        # self.V_stance = nn.Parameter(torch.FloatTensor(self.dim, 1))
        # nn.init.xavier_uniform_(self.V_stance.data, gain=1)
        # #self.bias_stance = nn.Parameter(torch.FloatTensor(100))
        # self.bias_stance = nn.Parameter(torch.FloatTensor(3, self.dim))

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))
        #self.bias_ts1 = nn.Parameter(torch.FloatTensor(100))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)
        #self.sequence_stance = torch.nn.Linear(300, 3)
        #self.argument = torch.nn.Linear(768, 2)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.crf_ts1 = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        #self.mask = torch.tensor(np.array([[1., 0, 0, 0, 1., 1.]]), dtype=torch.float32)
        #self.mask2 = torch.tensor(np.array([[1., 0, 0], [0, 1., 1.]]), dtype=torch.float32)
        #self.trans = torch.tensor(np.array([[1., 0, 0]]), dtype=torch.float32)
        #self.attention = self_attention_layer(768)
        #self.attention2 = self_attention_layer2(3)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            ######stm_lm_hs = self.stm_lm(ote_hs)
            #########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(ote_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            #ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            #########stm_lm_hs = self.stm_lm(ote_hs)
            ###########stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            #p_y_x_ote = self.fc_ote(ote_hs)
            ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            # 边界的基础上判断情感


            ##########stm_lm_ts = self.stm_ts(ts_hs)
            ##########stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(ts_hs)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                #loss_ts = -self.crf_ts(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts1(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts1.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_0312_layer1_idcnn_crf_x(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_0312_layer1_idcnn_crf_x, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
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


        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        #self.drnn_start_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.drnn_end_linear = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        #self.W_gate = torch.nn.init.xavier_uniform_(torch.empty(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.id_cnn_ote = torch.nn.Linear(300, 2)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        self.id_cnn_ts = torch.nn.Linear(300, 3)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)

        self.initial_filter_width = 3
        self.initial_padding = 1
        self.dilation = [1, 2, 1]
        self.padding = [1, 2, 1]
        initial_filter_width = self.initial_filter_width
        initial_num_filters = 300
        self.itdicnn0 = nn.Conv1d(768, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.itdicnn2 = nn.Conv1d(300, initial_num_filters, kernel_size=initial_filter_width,
                                  padding=self.initial_padding, bias=True)
        self.itdicnn21 = nn.ModuleList([nn.Conv1d(initial_num_filters, initial_num_filters,
                                                kernel_size=initial_filter_width, padding=self.padding[i],
                                                dilation=self.dilation[i], bias=True) for i in
                                      range(0, len(self.padding))])
        self.repeats = 1
        self.take_layer = [False, False, True]
        self.layer_residual = True
        self.block_residual = True
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output = F.relu(self.itdicnn0(sequence_output))
            # print(inner_last_output.shape)
            last_dims = 768
            last_output = inner_last_output
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores = []
            for block in range(self.repeats):

                hidden_outputs = []
                total_output_width = 0
                inner_last_dims = last_dims
                inner_last_output = last_output
                block_input = last_output
                for layeri, conv in enumerate(self.itdicnn):
                    h = F.relu(conv(inner_last_output))
                    if self.take_layer[layeri]:
                        hidden_outputs.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output = h + inner_last_output
                    else:
                        inner_last_output = h

                h_concat = torch.cat(hidden_outputs, -1)
                if self.block_residual:
                    last_output = self.dropout(h_concat) + block_input
                else:
                    last_output = self.dropout(h_concat)
                last_dims = total_output_width

                block_unflat_scores.append(last_output)
            p_y_x_ote = self.id_cnn_ote(block_unflat_scores[0].permute(0, 2, 1))

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)

            sequence_output = sequence_output.permute(0, 2, 1)
            inner_last_output2 = F.relu(self.itdicnn0(sequence_output))

            # print(inner_last_output.shape)
            last_dims2 = 768
            last_output2 = inner_last_output2
            # There could be output mismatch here, please consider concatenation if so
            block_unflat_scores2 = []
            for block in range(self.repeats):

                hidden_outputs2 = []
                total_output_width = 0
                inner_last_dims = last_dims2
                inner_last_output2 = last_output2
                block_input2 = last_output2
                for layeri, conv in enumerate(self.itdicnn21):
                    h = F.relu(conv(inner_last_output2))
                    if self.take_layer[layeri]:
                        hidden_outputs2.append(h)
                        total_output_width += 768
                    inner_last_dims = 768
                    if self.layer_residual:
                        inner_last_output2 = h + inner_last_output2
                    else:
                        inner_last_output2 = h

                h_concat2 = torch.cat(hidden_outputs2, -1)
                if self.block_residual:
                    last_output2 = self.dropout(h_concat2) + block_input2
                else:
                    last_output2 = self.dropout(h_concat2)
                last_dims2 = total_output_width

                block_unflat_scores2.append(last_output2)
            p_y_x_ts = self.id_cnn_ts(block_unflat_scores2[0].permute(0, 2, 1))

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2,
                                                                                                                    3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # loss_ote = loss_fct(
                #     p_y_x_ote.view(-1, 2),
                #     IO_labels.view(-1)
                # )
                # loss_ts = loss_fct(
                #     p_y_x_ts.view(-1, 3),
                #     labels.view(-1)
                # )
                # loss_ts1 = loss_fct(
                #     p_y_x_ts_tilde.view(-1, 3),
                #     labels.view(-1)
                # )
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                # loss_ts = -self.crf_ote(p_y_x_ts, labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_Li(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_Li, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ts = nn.GRU(input_size=768,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            # stm_lm_hs = self.stm_lm(ote_hs)
            # ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13
            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())


class Token_pipeline_Li_B1C0(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_Li_B1C0, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ts = nn.GRU(input_size=768,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            a=torch.Tensor(mask_positions).view(90, -1).expand(90,768).cuda()
            sequence_output_mask = sequence_output.view(90, 768)*a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            ote_stance = self.stm_lm(ote_stance)
            #长度限制下次加
            topic_ = self.embed(topic_indices)
            topic_ = topic_.reshape(topic_.shape[0], 1, topic_.shape[1])
            topic_ = topic_.expand(topic_.shape[0], ote_stance.shape[1], topic_.shape[2])
            alpha = torch.matmul(F.tanh(torch.matmul(torch.cat((topic_, ote_stance), 2), self.W_ts1) + self.bias_ts1), self.V_ts1).transpose(1, 2)
            alpha = F.softmax(alpha.sum(1, keepdim=True), dim=2)
            x = torch.matmul(alpha, ote_stance).squeeze(1)  # batch_size x 2*hidden_dim
            stance = self.stance(x)
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return  torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            # stm_lm_hs = self.stm_lm(ote_hs)
            # ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

            p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###


            if labels is not None:
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())

class Token_pipeline_Li_B0C1(nn.Module):#注意力
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)q
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(Token_pipeline_Li_B0C1, self).__init__()

        self.num_labels = 3
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.hidden_size = 300
        self.dim_ote_h = 150
        self.dim_ote_y = 2
        self.dim_ts_y = 3
        self.bert = BertModel.from_pretrained(model_name)
        self.max_len = 90

        self.gru_ote = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ote_stance = nn.GRU(input_size=768,
                              hidden_size=self.dim_ote_h,
                              num_layers=1,
                              bias=True,
                              dropout=0,
                              batch_first=True,
                              bidirectional=True)
        self.gru_ts = nn.GRU(input_size=768,
                             hidden_size=self.dim_ote_h,
                             num_layers=1,
                             bias=True,
                             dropout=0,
                             batch_first=True,
                             bidirectional=True)

        self.stm_lm = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_ts = torch.nn.Linear(2 * self.dim_ote_h, 2 * self.dim_ote_h)
        self.stm_lm_activation = nn.Tanh()
        self.fc_stm = torch.nn.Linear(2 * self.dim_ote_h, 2)
        self.W_gate = nn.Parameter(torch.FloatTensor(2 * self.dim_ote_h, 2 * self.dim_ote_h))
        nn.init.xavier_uniform_(self.W_gate.data, gain=1)
        self.W_gate2 = nn.Parameter(torch.FloatTensor(2 * self.max_len, 3 * self.max_len))
        nn.init.xavier_uniform_(self.W_gate2.data, gain=1)

        self.dim = 100

        self.W_ts1 = nn.Parameter(torch.FloatTensor(600, 300))
        nn.init.xavier_uniform_(self.W_ts1.data, gain=1)
        self.V_ts1 = nn.Parameter(torch.FloatTensor(300, 1))
        nn.init.xavier_uniform_(self.V_ts1.data, gain=1)
        self.bias_ts1 = nn.Parameter(torch.FloatTensor(300))

        self.W_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 3, self.dim))
        nn.init.xavier_uniform_(self.W_ts2.data, gain=1)
        self.V_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, self.dim, 1))
        nn.init.xavier_uniform_(self.V_ts2.data, gain=1)
        self.bias_ts2 = nn.Parameter(torch.FloatTensor(self.max_len, 2, self.dim))
        #self.bias_ts2 = nn.Parameter(torch.FloatTensor(100))

        self.dropout = nn.Dropout(0.3)

        self.stance = torch.nn.Linear(300, 3)
        self.stance_gat = torch.nn.Linear(768, 3)

        self.fc_ote = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ote_y)
        self.fc_ts = torch.nn.Linear(2 * self.dim_ote_h, self.dim_ts_y)
        # if self.use_crf:
        self.crf_ote = CRF(2, batch_first=self.batch_first)

        self.crf_ts = CRF(3, batch_first=self.batch_first)
        self.stance2argument = torch.tensor(np.array([[1., 0.], [0., 1.], [0., 1.]]), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
    def forward(self, md, input_ids, attention_mask, token_type_ids, argument_input_ids, argument_attention_mask,
                argument_token_type_ids, labels=None, IO_labels=None, stance_label=None, argument_label=None,mask_positions=None,topic_indices=None,label_stance=None, trans_matrix=None):
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        outputs = self.bert(
            argument_input_ids,
            attention_mask=argument_attention_mask,
            token_type_ids=argument_token_type_ids
        )
        sequence_output = outputs[0]
        stance_cls_output = outputs[1]

        if md == 'IO_inference':
            sequence_output = self.dropout(sequence_output)
            ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            stm_lm_hs = self.stm_lm(ote_hs)
            ###stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            p_y_x_ote = self.fc_ote(stm_lm_hs)

            if labels is not None:
                loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                return loss_ote
            else:  # inference
                return self.crf_ote.decode(p_y_x_ote, attention_mask.byte())
        if md == 'IO_stance_inference':
            #长度限制下次加
            a = torch.Tensor(mask_positions).view(90, -1).expand(90, 768).cuda()
            sequence_output_mask = sequence_output.view(90, 768) * a
            ote_stance, final = self.gru_ote_stance(sequence_output_mask.view(sequence_output.shape[0], -1, 768))
            forward = final[0].view(1, -1)
            backward = final[1].view(1, -1)
            stance = self.stance(torch.cat((forward, backward), 1))
            stance_softmax = F.softmax(stance, dim=1)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                stance_loss = loss_fct(
                    stance.view(-1, 3),
                    torch.LongTensor(label_stance).view(-1).cuda()
                )
                return stance_loss, stance_softmax
            else:  # inference
                return torch.argmax(stance, dim=1), stance_softmax
        if md == 'unified_inference':
            sequence_output = self.dropout(sequence_output)
            #ote_hs, _ = self.gru_ote(sequence_output.view(sequence_output.shape[0], -1, 768))
            # stm_lm_hs = self.stm_lm(ote_hs)
            # ####stm_lm_hs = self.dropout(stm_lm_hs)  # 只判断边界
            # p_y_x_ote = self.fc_ote(stm_lm_hs)
            ts_hs, _ = self.gru_ts(sequence_output.view(sequence_output.shape[0], -1, 768))
            # 边界的基础上判断情感

            ts_hs_transpose = torch.transpose(ts_hs, 0, 1)
            ts_hs_tilde = []
            h_tilde_tm1 = object
            for i in range(ts_hs_transpose.shape[0]):
                if i == 0:
                    h_tilde_t = ts_hs_transpose[i]
                    # ts_hs_tilde = h_tilde_t.view(1, -1)
                else:
                    # t-th hidden state for the task targeted sentiment
                    ts_ht = ts_hs_transpose[i]
                    gt = torch.sigmoid(torch.mm(ts_ht, self.W_gate.to('cuda')))
                    # a= (1 - gt)
                    h_tilde_t = gt * ts_ht + (1 - gt) * h_tilde_tm1
                    # print(h_tilde_t)
                # ts_hs_tilde.append(h_tilde_t.view(1,-1))
                if i == 0:
                    ts_hs_tilde = h_tilde_t.unsqueeze(0)
                else:
                    ts_hs_tilde = torch.cat((ts_hs_tilde, h_tilde_t.unsqueeze(0)), 0)
                # ts_hs_tildes = torch.cat((ts_hs_tildes, ts_hs_tilde.unsqueeze(0)), 0)

                h_tilde_tm1 = h_tilde_t
            ts_hs = torch.transpose(ts_hs_tilde, 0, 1)
            stm_lm_ts = self.stm_ts(ts_hs)
            ####stm_lm_ts = self.dropout(stm_lm_ts)

            ote2ts = torch.Tensor(trans_matrix).view(90, -1).cuda()
            p_y_x_ts = self.fc_ts(stm_lm_ts)
            p_y_x_ts_softmax = F.softmax(p_y_x_ts, dim=-1)  # 13

            # p_y_x_ts_tilde = ote2ts + p_y_x_ts_softmax
            # p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            a = torch.cat((ote2ts.unsqueeze(0).unsqueeze(2), p_y_x_ts_softmax.unsqueeze(2)), 2)
            alpha_ts1 = torch.matmul(torch.tanh(torch.matmul(a, self.W_ts2) + self.bias_ts2), self.V_ts2).transpose(2, 3)
            alpha_ts1 = F.softmax(alpha_ts1.sum(2, keepdim=True), dim=3)
            p_y_x_ts_tilde = torch.matmul(alpha_ts1, a)
            p_y_x_ts_tilde = p_y_x_ts_tilde.contiguous().view(-1, self.max_len, 3)  ###

            if labels is not None:
                #loss_ote = -self.crf_ote(p_y_x_ote, IO_labels, attention_mask.byte(), reduction='token_mean')
                loss_ts1 = -self.crf_ts(p_y_x_ts_tilde, labels, attention_mask.byte(), reduction='token_mean')

                return loss_ts1
            else:  # inference
                return self.crf_ts.decode(p_y_x_ts_tilde, attention_mask.byte())
