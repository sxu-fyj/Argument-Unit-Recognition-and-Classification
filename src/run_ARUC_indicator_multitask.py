#!/usr/bin/env python

import os
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import datetime as dt

from sty import fg, bg, ef, rs

# import spacy
# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_sm")
# from spacy.gold import align

from sklearn import metrics

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)

from transformers import (AdamW, BertConfig, BertTokenizer,
                          get_linear_schedule_with_warmup,
                          BertModel, BertPreTrainedModel)

from utils import InputFeatures_indicator, get_data_with_labels

from models2 import Token_indicator_multitask


def training(train_dataloader, model, device, optimizer, scheduler, max_grad_norm):
    model.train()
    total_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    predictions_train, true_labels_train = [], []
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_sentence_ids, batch_label_ids, batch_BIEO_label_ids, batch_BIEOstance_label_ids, batch_stance_label_ids, batch_argument_label_ids = batch
        # print(batch_input_ids.shape[0])
        # forward pass
        loss, splits = model(
            batch_input_ids,
            token_type_ids=batch_sentence_ids,
            attention_mask=batch_input_mask,
            labels=batch_label_ids,
            BIEO_labels=batch_BIEO_label_ids,
            BIEO_stance_labels=batch_BIEOstance_label_ids,
            stance_label=batch_stance_label_ids,
            argument_label=batch_argument_label_ids
            # batch_size = batch_input_ids.shape[0]
        )

        loss_ts, loss_ote, stance_loss, argument_loss, cos = splits[0], splits[1], splits[2], splits[3], splits[4]
        # backward pass
        if step%10==0:
            print(loss_ts)
            print(loss_ote)
            print(stance_loss)
            print(argument_loss)
            print(cos)
            print('-----------------------------------------')
        loss.backward()

        # track train loss
        total_loss += loss.item()
        nb_tr_examples += batch_input_ids.size(0)
        nb_tr_steps += 1

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # update learning rate
        scheduler.step()

        model.zero_grad()

    return model, optimizer, scheduler, total_loss


def evaluation(sample_dataloader, model, device, tokenizer):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    for step, batch in enumerate(sample_dataloader):
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_sentence_ids, batch_label_ids, batch_BIEO_label_ids, batch_BIEOstance_label_ids, batch_stance_label_ids, batch_argument_label_ids = batch

        id2BIEO = {0: 'O', 1: 'B-PRO', 2: 'I-PRO', 3: 'E-PRO', 4: 'B-CON', 5: 'I-CON', 6: 'E-CON'}
        BIEO2ids = {'O': 0, 'B-PRO': 2, 'I-PRO': 2, 'E-PRO': 2, 'B-CON': 1, 'I-CON': 1, 'E-CON': 1}

        with torch.no_grad():
            tagssss = model(
                batch_input_ids,
                token_type_ids=batch_sentence_ids,
                attention_mask=batch_input_mask)
        batch_bieo_stance_tags = tagssss[1]
        batch_correct_label_ids = []
        batch_correct_tags = []
        for input_ids, label_ids, tags in zip(batch_input_ids, batch_label_ids, batch_bieo_stance_tags):
            input_ids = input_ids.cpu().tolist()
            label_ids = label_ids.cpu().tolist()
            #if type(tags) != list:
            #tags = tags.cpu().tolist()
            seq_len = [i for i, t in enumerate(input_ids) if t == 102][0]  # 102 is the [SEP] token_id
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            correct_label_ids = [l for t, l in zip(input_tokens[1:seq_len], label_ids[1:seq_len]) if
                                 not t.startswith('##')]

            tags = [BIEO2ids[id2BIEO[l]] for l in tags]
            correct_tags = [l for t, l in zip(input_tokens[1:seq_len], tags[1:seq_len]) if not t.startswith('##')]
            seq = " ".join(input_tokens[1:seq_len]).replace(' ##', '')
            assert len(correct_label_ids) == len(seq.split(' '))
            batch_correct_label_ids.append(correct_label_ids)  # 真实值
            batch_correct_tags.append(correct_tags)  # 预测值
        y_true += batch_correct_label_ids
        y_pred += batch_correct_tags

    # flatten
    YT, YP = [], []
    for t, p in zip(y_true, y_pred):
        YT += t
        YP += p
    assert len(YT) == len(YP)

    p, r, f1s, s = metrics.precision_recall_fscore_support(y_pred=YP, y_true=YT, average='macro', warn_for=tuple())
    return y_true, y_pred, p, r, f1s


def ot2bieos_ote(ote_tag_sequence):
    """
    ot2bieos function for ote task
    :param ote_tag_sequence:
    :return:
    """
    n_tags = len(ote_tag_sequence)
    prev_ote_tag = '$$$'

    new_ote_sequence_ids = []
    new_ote_sequence_label = []
    BIEO2ids = {'O': 0, 'B': 1, 'I': 2, 'E': 3}
    id2BIEO = {0: 'O', 1: 'B', 2: 'I', 3: 'E'}
    for i in range(n_tags):
        cur_ote_tag = ote_tag_sequence[i]
        if cur_ote_tag == 0:
            new_ote_sequence_ids.append(0)
            new_ote_sequence_label.append('O')
        else:
            # cur_ote_tag is T
            if prev_ote_tag != cur_ote_tag:
                # prev_ote_tag is O, new_cur_tag can only be B or S
                if i == n_tags - 1:
                    new_ote_sequence_ids.append(4)
                    new_ote_sequence_label.append('S')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    new_ote_sequence_ids.append(1)
                    new_ote_sequence_label.append('B')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    new_ote_sequence_ids.append(4)
                    new_ote_sequence_label.append('S')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
            else:
                # prev_tag is T, new_cur_tag can only be I or E
                if i == n_tags - 1:
                    new_ote_sequence_ids.append(3)
                    new_ote_sequence_label.append('E')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    # next_tag is T
                    new_ote_sequence_ids.append(2)
                    new_ote_sequence_label.append('I')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    # next_tag is O
                    new_ote_sequence_ids.append(3)
                    new_ote_sequence_label.append('E')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
        prev_ote_tag = cur_ote_tag
    return new_ote_sequence_ids, new_ote_sequence_label


def ot2bieos_ts(ts_tag_sequence, s):
    """
    ot2bieos function for ts task
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    prev_pos = '$$$'
    new_ote_sequence_ids = []
    new_ote_sequence_label = []
    BIEO2ids = {'O': 0, 'B-PRO': 1, 'I-PRO': 2, 'E-PRO': 3, 'B-CON': 4, 'I-CON': 5, 'E-CON': 6}
    id2BIEO = {0: 'O', 1: 'B-PRO', 2: 'I-PRO', 3: 'E-PRO', 4: 'B-CON', 5: 'I-CON', 6: 'E-CON'}
    id2stance={1: 'CON', 2: 'PRO'}
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 0:
            new_ote_sequence_ids.append(0)
            new_ote_sequence_label.append('O')
            cur_pos = 0
        else:
            cur_pos = cur_ts_tag
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ote_sequence_ids.append(BIEO2ids['S-%s' % id2stance[cur_pos]])
                    new_ote_sequence_label.append('S-%s' % id2stance[cur_pos])
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 0:
                        new_ote_sequence_ids.append(BIEO2ids['S-%s' % id2stance[cur_pos]])
                        new_ote_sequence_label.append('S-%s' % id2stance[cur_pos])
                    else:
                        new_ote_sequence_ids.append(BIEO2ids['B-%s' % id2stance[cur_pos]])
                        new_ote_sequence_label.append('B-%s' % id2stance[cur_pos])
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_ote_sequence_ids.append(BIEO2ids['E-%s' % id2stance[cur_pos]])
                    new_ote_sequence_label.append('E-%s' % id2stance[cur_pos])
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 0:
                        new_ote_sequence_ids.append(BIEO2ids['E-%s' % id2stance[cur_pos]])
                        new_ote_sequence_label.append('E-%s' % id2stance[cur_pos])
                    else:
                        new_ote_sequence_ids.append(BIEO2ids['I-%s' % id2stance[cur_pos]])
                        new_ote_sequence_label.append('I-%s' % id2stance[cur_pos])
        prev_pos = cur_pos
    return new_ote_sequence_ids, new_ote_sequence_label

def main():
    parser = argparse.ArgumentParser(description='Run Fine-Grained Argument Unit Recognition and Classification.')
    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    # 随机种子
    parser.add_argument('--card_number', type=int, default=0, help='Your GPU card number.')
    # GPU
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs for training.')
    #
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels; either 3 (pro, con, non) or 2 (arg, non).')
    # 标签数量
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation.')
    # ?
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='The learning rate.')
    # 学习率
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The max_grad_norm.')
    # ?
    parser.add_argument('--max_sequence_length', type=int, default=100,
                        help='The maximum sequence length of the input text.')
    # 最大长度
    parser.add_argument('--train_batch_size', type=int, default=64, help='Your train set batch size.')
    # batch_size
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Your evaluation set batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Your test set batch size.')
    parser.add_argument('--target_domain', type=str, default='In-Domain', help='In-Domain OR Cross-Domain')
    # 域内和跨域两个任务分支
    parser.add_argument('--input_file', type=str, default='../data/AURC_DATA_dict_se.json', help='The input dict file.')
    # 预处理好的数据
    parser.add_argument('--data_dir', type=str, default='../data/', help='The data directory.')
    # 数据位置
    parser.add_argument('--output_dir', type=str, default='../models/',
                        help='The output directory of the model, config and predictions.')
    # 输出模型的位置
    parser.add_argument('--pretrained_weights', type=str, default='bert-large-cased-whole-word-masking',
                        help='The pretrained bert model.')
    # bert预训练向量
    parser.add_argument("--fine_tuning", default=True, action="store_true", help="Flag for full fine-tuning.")
    # 微调
    parser.add_argument("--crf", default=False, action="store_true", help="Flag for CRF useage.")
    # 是否使用crf
    parser.add_argument("--train", default=False, action="store_true", help="Flag for training.")
    # 可选
    parser.add_argument("--eval", default=True, action="store_true", help="Flag for evaluation.")
    # 可选
    parser.add_argument("--save_model", default=True, action="store_true", help="Flag for saving.")
    # 保存可选
    parser.add_argument("--save_prediction", default=True, action="store_true", help="Flag for saving.")
    # 预测保存,可选
    args = parser.parse_args()

    if args.seed is not None:
        import numpy
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True  # 如果想要避免这种结果波动，设置True
        torch.backends.cudnn.benchmark = False
    #############################################################################
    # Parameters and paths
    device = torch.device("cuda:{}".format(args.card_number) if torch.cuda.is_available() else "cpu")
    print("device", device)

    task = '_'.join(['aurc', args.target_domain[:2].lower()])

    MODEL_PATH = os.path.join(args.output_dir, '{}_token.pt'.format(task))
    print(MODEL_PATH)
    CONFIG_PATH = os.path.join(args.output_dir, '{}_config.json'.format(task))
    print(CONFIG_PATH)
    PREDICTIONS_DEV = os.path.join(args.output_dir, '{}_predictions_dev.json'.format(task))
    print(PREDICTIONS_DEV)
    PREDICTIONS_TEST = os.path.join(args.output_dir, '{}_predictions_test.json'.format(task))
    print(PREDICTIONS_TEST)

    #############################################################################
    # Load Data
    fname = os.path.join(args.data_dir, args.input_file)
    print(fname)
    with open(fname, 'r') as my_file:
        AURC_DATA_dict = json.load(my_file)
    print(len(AURC_DATA_dict), [len(AURC_DATA_dict[topic]) for topic in AURC_DATA_dict.keys()])

    topics = sorted(set(AURC_DATA_dict.keys()))
    print(len(topics), topics)

    label2id = dict()
    label2id['non'] = 0
    if args.num_labels == 2:
        label2id['arg'] = 1
    else:
        assert args.num_labels == 3
        label2id['con'] = 1
        label2id['pro'] = 2

    stance_label2id = dict()
    stance_label2id['non'] = 0
    stance_label2id['con'] = 1
    stance_label2id['pro'] = 2

    #argument_label2id = dict()
    #stance_label2id['non'] = 0
    #stance_label2id['non'] = 1

    tokenizer = BertTokenizer.from_pretrained('../model_base/')
    # tokenizer = BertTokenizer.from_pretrained(args.pretrained_weights)

    train_features = []
    eval_features = []
    test_features = []

    all_input_tokens_dev = []
    all_input_tokens_test = []
    max_1 = 0
    for topic, AD in AURC_DATA_dict.items():
        for ad in AD:
            sequence_dict = tokenizer.encode_plus(ad['sentence'], max_length=args.max_sequence_length,
                                                  pad_to_max_length=True, add_special_tokens=True)
            # input_labels = [label2id[label] for label in ad['tokenized_sentence_bert_labels'].split(' ')]
            input_labels = [label2id[label] for label in ad['tokenized_sentence_bert_single_labels'].split(' ')]
            input_labels = [0] + input_labels[:args.max_sequence_length - 1] + [0] * max(0,
                                                                                         args.max_sequence_length - len(
                                                                                             input_labels) - 1)
            sequence_dict['label_ids'] = input_labels
            sequence_dict['BIEO_label_ids'], sequence_dict['BIEO_label'] = ot2bieos_ote(input_labels)
            sequence_dict['BIEO-stance_label_ids'], sequence_dict['BIEO-stance_label'] = ot2bieos_ts(input_labels,ad['sentence'])
            sequence_dict['input_tokens'] = tokenizer.convert_ids_to_tokens(sequence_dict['input_ids'])

            # start_positions = [0] * len(input_labels)
            # end_positions = [0] * len(input_labels)
            # if ad['is_argument'] == 1:
            #     start_positions[ad['start'] + 1] = 1
            #     end_positions[ad['end'] + 1] = 1
            # elif ad['is_argument'] == 0:
            #     start_positions[0] = 1
            #     end_positions[0] = 1
            # sequence_dict['start_ids'] = start_positions
            # sequence_dict['end_ids'] = end_positions


            for k, v in sequence_dict.items():
                assert len(v) == args.max_sequence_length
            sequence_dict['stance_ids'] = stance_label2id[ad['sentence_level_stance']]
            sequence_dict['argument_ids'] = ad['is_argument']

            FE = [
                InputFeatures_indicator(
                    input_ids = sequence_dict['input_ids'],
                    attention_mask = sequence_dict['attention_mask'],
                    token_type_ids = sequence_dict['token_type_ids'],
                    label_ids = sequence_dict['label_ids'],
                    BIEO_label_ids = sequence_dict['BIEO_label_ids'],
                    BIEO_label = sequence_dict['BIEO_label'],
                    BIEOstance_label_ids = sequence_dict['BIEO-stance_label_ids'],
                    BIEOstance_label = sequence_dict['BIEO-stance_label'],
                    stance_label = sequence_dict['stance_ids'],
                    argument_label = sequence_dict['argument_ids'],
                    sentence_hash=ad['sentence_hash']
                )
            ]
            if len(ad['tokenized_sentence_bert'].split()) > max_1:
                max_1 = len(ad['tokenized_sentence_bert'].split())
                print(len(ad['tokenized_sentence_bert'].split()))
                print(ad['tokenized_sentence_bert'])
            if ad[args.target_domain] == 'Train':
                train_features += FE
            if ad[args.target_domain] == 'Dev':
                eval_features += FE
                all_input_tokens_dev.append(tokenizer.convert_ids_to_tokens(sequence_dict['input_ids']))
            if ad[args.target_domain] == 'Test':
                test_features += FE
                all_input_tokens_test.append(tokenizer.convert_ids_to_tokens(sequence_dict['input_ids']))
    # TRAIN DATA
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    all_BIEO_label_ids = torch.tensor([f.BIEO_label_ids for f in train_features], dtype=torch.long)
    all_BIEOstance_label_ids = torch.tensor([f.BIEOstance_label_ids for f in train_features], dtype=torch.long)
    all_stance_label_ids = torch.tensor([f.stance_label for f in train_features], dtype=torch.long)
    all_argument_label_ids = torch.tensor([f.argument_label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_BIEO_label_ids, all_BIEOstance_label_ids, all_stance_label_ids, all_argument_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    print(len(train_sampler), len(train_dataloader))

    # EVAL DATA
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    all_BIEO_label_ids = torch.tensor([f.BIEO_label_ids for f in eval_features], dtype=torch.long)
    all_BIEOstance_label_ids = torch.tensor([f.BIEOstance_label_ids for f in eval_features], dtype=torch.long)
    all_stance_label_ids = torch.tensor([f.stance_label for f in eval_features], dtype=torch.long)
    all_argument_label_ids = torch.tensor([f.argument_label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_BIEO_label_ids, all_BIEOstance_label_ids, all_stance_label_ids, all_argument_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    print(len(eval_sampler), len(eval_dataloader))

    # TEST DATA
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)
    all_BIEO_label_ids = torch.tensor([f.BIEO_label_ids for f in test_features], dtype=torch.long)
    all_BIEOstance_label_ids = torch.tensor([f.BIEOstance_label_ids for f in test_features], dtype=torch.long)
    all_stance_label_ids = torch.tensor([f.stance_label for f in test_features], dtype=torch.long)
    all_argument_label_ids = torch.tensor([f.argument_label for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_BIEO_label_ids, all_BIEOstance_label_ids, all_stance_label_ids, all_argument_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
    print(len(test_sampler), len(test_dataloader))

    #############################################################################
    # Training
    if args.train:

        # Load Config
        config = BertConfig.from_pretrained('../model_base/', num_labels=args.num_labels)
        # config = BertConfig.from_pretrained(args.pretrained_weights, num_labels=args.num_labels)
        config.hidden_dropout_prob = 0.1

        # Model
        model = Token_indicator_multitask(
            model_name='../model_base/',
            # model_name=args.pretrained_weights,
            num_labels=args.num_labels,
            output_hidden_states=False,
            use_crf=args.crf)
        model.to(device)

        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs
        print(num_train_optimization_steps)

        if args.fine_tuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.tokenbert.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=num_train_optimization_steps)

        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        # Main Loop
        print("##### DOMAIN:", args.target_domain, ",", "use CRF:", args.crf, ",", "learning-rate:", args.learning_rate,
              ",", "DROPOUT:", config.hidden_dropout_prob)
        print()

        best_p_dev, best_r_dev, best_f1s_dev = 0.0, 0.0, 0.0
        best_epoch = None
        prev_tr_loss = 1e8

        best_y_true_dev, best_y_pred_dev = [], []

        for epoch in range(args.epochs):
            print("Epoch: %4i" % epoch, dt.datetime.now())

            # TRAINING
            model, optimizer, scheduler, tr_loss = training(train_dataloader, model=model, device=device,
                                                            optimizer=optimizer, scheduler=scheduler,
                                                            max_grad_norm=args.max_grad_norm)

            # EVALUATION: TRAIN SET
            y_true_train, y_pred_train, p_train, r_train, f1s_train = evaluation(
                train_dataloader, model=model, device=device, tokenizer=tokenizer)
            print("TRAIN:  Pre. %.3f | Rec. %.3f | F1 %.3f" % (p_train, r_train, f1s_train))

            # EVALUATION: DEV SET
            y_true_dev, y_pred_dev, p_dev, r_dev, f1s_dev = evaluation(
                eval_dataloader, model=model, device=device, tokenizer=tokenizer)
            print("EVAL:   Pre. %.3f | Rec. %.3f | F1 %.3f | BEST F1: %.3f" % (p_dev, r_dev, f1s_dev, best_f1s_dev),
                  best_epoch)

            if f1s_dev > best_f1s_dev:
                best_f1s_dev = f1s_dev
                best_epoch = epoch

                # EVALUATION: TEST SET
                y_true_test, y_pred_test, p_test, r_test, f1s_test = evaluation(
                    test_dataloader, model=model, device=device, tokenizer=tokenizer)
                print("TEST:   Pre. %.3f | Rec. %.3f | F1 %.3f" % (p_test, r_test, f1s_test))
                if args.save_model:
                    # Save Model
                    torch.save(model.state_dict(), MODEL_PATH)
                    # Save Config
                    with open(CONFIG_PATH, 'w') as f:
                        json.dump(config.to_json_string(), f, sort_keys=True, indent=4, separators=(',', ': '))

                if args.save_prediction:
                    # SAVE PREDICTED DATA
                    # DEV
                    # all_input_tokens_dev=tokenizer.convert_ids_to_tokens(batch_input_ids_dev)
                    # all_input_tokens_test=tokenizer.convert_ids_to_tokens(batch_input_ids_test)
                    DATA_DEV = get_data_with_labels(all_input_tokens_dev, y_true_dev, y_pred_dev)
                    with open(PREDICTIONS_DEV, 'w') as f:
                        json.dump(DATA_DEV, f, sort_keys=True, indent=4, separators=(',', ': '))
                    # TEST
                    DATA_TEST = get_data_with_labels(all_input_tokens_test, y_true_test, y_pred_test)
                    with open(PREDICTIONS_TEST, 'w') as f:
                        json.dump(DATA_TEST, f, sort_keys=True, indent=4, separators=(',', ': '))
            print()

    #################################################################################
    # Load the fine-tuned model:
    if args.eval:
        # Model
        model = Token_indicator_multitask(
            model_name='../model_base/',
            #            model_name=args.pretrained_weights,
            num_labels=args.num_labels,
            output_hidden_states=False,
            use_crf=args.crf)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)

        print("\nInference:\n")

        # EVALUATION: TRAIN SET
        y_true_train, y_pred_train, p_train, r_train, f1s_train = evaluation(
            train_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("TRAIN:  Pre. %.3f | Rec. %.3f | F1 %.3f" % (p_train, r_train, f1s_train))

        # EVALUATION: DEV SET
        y_true_dev, y_pred_dev, p_dev, r_dev, f1s_dev = evaluation(
            eval_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("EVAL:   Pre. %.3f | Rec. %.3f | F1 %.3f" % (p_dev, r_dev, f1s_dev))

        # EVALUATION: TEST SET
        y_true_test, y_pred_test, p_test, r_test, f1s_test = evaluation(
            test_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("TEST:   Pre. %.3f | Rec. %.3f | F1 %.3f" % (p_test, r_test, f1s_test))


if __name__ == "__main__":
    main()

