#!/usr/bin/env python

import torch
from torch import nn

from transformers import BertForTokenClassification
from transformers import BertModel

from torchcrf import CRF

class TokenBERT(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(TokenBERT, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        self.rnn = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.tokenbert.dropout(sequence_output)  # (B, L, H)
        #sequence_output, _ = self.rnn(sequence_output.view(sequence_output.shape[0], -1, 768))

        logits = self.tokenbert.classifier(sequence_output)

        if self.use_crf:
            if labels is not None: # training
                return -self.crf(logits, labels, attention_mask.byte())
            else: # inference
                return self.crf.decode(logits, attention_mask.byte())
        else:
            if labels is not None: # training
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1)
                )
                return loss
            else: # inference
                return torch.argmax(logits, dim=2)


class TokenBERT_stance(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, batch_size=1):
        super(TokenBERT_stance, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        #self.batch_size=batch_size
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        self.bert = BertModel.from_pretrained(model_name)
        self.rnn = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.W_alpha = torch.nn.Parameter(torch.FloatTensor(768, 768))
        self.b_alpha = torch.nn.Parameter(torch.FloatTensor(10, 64))
        self.tanh_activation = nn.Tanh()

        self.fc_linear = torch.nn.Linear(768, 3)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, sentence_mask, topic_input_ids, topic_attention_mask, topic_token_type_ids, topic_mask,  labels=None, stance_label=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sentence_output = outputs[0]
        sentence_output = self.tokenbert.dropout(sentence_output)  # (B, L, H)
        #sequence_output, _ = self.rnn(sequence_output.view(sequence_output.shape[0], -1, 768))
        sentence_output = torch.mul(sentence_output, sentence_mask.unsqueeze(-1).repeat(1, 1, sentence_output.size(
            -1)))

        topic_all_encoder_layers = self.bert(
            topic_input_ids,
            attention_mask=topic_attention_mask,
            token_type_ids=topic_token_type_ids
        )
        # all_encoder_layers, cls_output = self.bert(input_ids, token_type_ids, attention_mask)
        topic_output = topic_all_encoder_layers[0]

        topic_output = self.dropout(topic_output)
        topic_output = torch.mul(topic_output, topic_mask.unsqueeze(-1).repeat(1, 1, topic_output.size(
            -1)))  # [batch, max_len, -1, dim] 将cls去掉

        alpha_matrix = self.tanh_activation(torch.matmul(torch.matmul(topic_output, self.W_alpha), torch.transpose(sentence_output,1,2))+self.b_alpha)
        alpha_ = torch.sum(alpha_matrix,dim = 1,keepdim=True)
        final = torch.matmul(alpha_, sentence_output)
        logits = self.fc_linear(final)
        #logits = logits.view(-1,3)
        if labels is not None: # training
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, 3),
                stance_label.view(-1)
            )
            return loss
        else: # inference
            return torch.argmax(logits, dim=2)

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    # print(log_probs[0])
    # print(positions[0])
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(
                                   dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss


class TokenBERT_span(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1, device='cpu'):
        super(TokenBERT_span, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        # self.batch_size=batch_size
        self.bert = BertModel.from_pretrained(model_name)

        self.rnn = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)
        #self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 300)
        self.dense_s = nn.Linear(768, 300)
        self.dense_e = nn.Linear(768, 300)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(300, 2)
        self.start_classifier = nn.Linear(300, 1)
        self.end_classifier = nn.Linear(300, 1)

        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                labels=None):
        #if mode == 'argument_inference':
        all_encoder_layers = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # print(start_positions[0])
        # print(end_positions[0])
        # all_encoder_layers, cls_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[0]
        #sequence_output = self.dropout(sequence_output)
        sequence_output_s = self.dense_s(sequence_output)
        sequence_output_s = self.activation(sequence_output_s)

        sequence_output_e = self.dense_e(sequence_output)
        sequence_output_e = self.activation(sequence_output_e)
        start_logits = self.start_classifier(sequence_output_s).squeeze(-1)  # [N, L, 2]
        end_logits = self.end_classifier(sequence_output_e).squeeze(-1)
        if start_positions is not None and end_positions is not None and labels is not None:
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            return start_loss, end_loss
        else:  # inference
            return torch.argmax(start_logits, dim=-1), torch.argmax(end_logits, dim=-1)


class TokenBERT_span2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
                 output_attentions=False, batch_first=True, use_crf=True, batch_size=1, device='cpu'):
        super(TokenBERT_span2, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        # self.batch_size=batch_size
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        self.rnn = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          bias=True,
                          dropout=0,
                          batch_first=True,
                          bidirectional=True)
        #self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 300)
        self.dense_s = nn.Linear(768, 300)
        self.dense_e = nn.Linear(768, 300)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(300, 2)
        self.start_classifier = nn.Linear(300, 1)
        self.end_classifier = nn.Linear(300, 1)

        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None,
                labels=None):
        #if mode == 'argument_inference':
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.tokenbert.dropout(sequence_output)
        #sequence_output = self.dropout(sequence_output)

        sequence_output_s = self.dense_s(sequence_output)
        sequence_output_s = self.activation(sequence_output_s)

        sequence_output_e = self.dense_e(sequence_output)
        sequence_output_e = self.activation(sequence_output_e)

        start_logits = self.start_classifier(sequence_output_s).squeeze(-1)  # [N, L, 2]
        end_logits = self.end_classifier(sequence_output_e).squeeze(-1)

        #logits = self.tokenbert.classifier(sequence_output)

        if start_positions is not None and end_positions is not None and labels is not None:
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            return start_loss, end_loss
        else:  # inference
            return torch.argmax(start_logits, dim=-1), torch.argmax(end_logits, dim=-1)
