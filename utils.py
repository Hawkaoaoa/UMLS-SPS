import numpy as np
import os
import random
import time
import logging

from scipy.stats import spearmanr
from sklearn import metrics
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


def smape(A, F):
    return 100/len(A) * sum(2 * abs(F - A) / (abs(A) + abs(F)))

def get_testing_strings(testing_set):

    all_strings = []
    for n in testing_set:
        the_string = n[0]
        all_strings.append(the_string)
    
    return all_strings


class BertSST2Model(nn.Module):

    def __init__(self, pretrained_name):
        """
        Args: 
            pretrained_name : bert model to be used
        """
        super(BertSST2Model, self).__init__()

        self.bert = AutoModel.from_pretrained(pretrained_name,
                                            return_dict=True)
        # ################
        # unfreeze_layers = ['layer.7', 'layer.8', 'layer.9', \
        #                             'layer.10','layer.11','bert.pooler','out.']

        # for name, param in self.bert.named_parameters():
        #         param.requires_grad = False
        #         for ele in unfreeze_layers:
        #             if ele in name:
        #                 param.requires_grad = True
        #                 break
        ################
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        # input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
        #     'token_type_ids'], inputs['attention_mask']
        # output = self.bert(input_ids, input_tyi, input_attn_mask)
        
        output = self.bert(**inputs)
        logits = self.fc(output.pooler_output)
        #####
        return self.sigmoid(logits)


def save_pretrained(model, optimizer, path):

    os.makedirs(path, exist_ok=True)
    
    # state = {'Coder': model.state_dict(), "AdamW": optimizer.state_dict()}
    state = {'Coder': model.state_dict()}
    torch.save(state, os.path.join(path, 'model.pth'))


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.dataset[index]


class collater():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        inputs, targets = [], []
        for item in examples:

            inputs.append(item[0])
            targets.append(float(item[1]))

        inputs = self.tokenizer(inputs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=100)
        targets = torch.tensor(targets)
        return inputs, targets


def train_epoch(optimizer, scheduler, criterion, train_loader, model, device, epoch, check_step):
    train_loss = 0
    train_cnt = 0
    batch_num = 0
    model.train()
    for train_iter, batch in enumerate(train_loader):

        batch_num += 1
        inputs, targets = [x.to(device) for x in batch]

        optimizer.zero_grad()

        bert_output = model(inputs)
        ###
        loss = criterion(bert_output, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_cnt += targets.shape[0]

        if train_iter % check_step == 0:
            logging.info('[%d, %5d] loss: %.5f' % (epoch, train_cnt, train_loss / train_cnt))


def test_epoch(criterion, test_loader, model, device, epoch):
    total_mae = 0
    val_cnt = 0
    output_lst = []
    targets_lst = []
    for val_iter, batch in enumerate(test_loader):
        inputs, targets = [x.to(device) for x in batch]

        model.eval()
        with torch.no_grad():
            
            bert_output = model(inputs)
            ###
            loss = criterion(bert_output, targets)
            total_mae += loss.item()
            val_cnt += targets.shape[0]

            output = bert_output.cpu().detach().numpy().flatten()
            targets = targets.cpu().detach().numpy().flatten()
            ###
            tmp_output_lst = output.tolist()
            output_lst += tmp_output_lst
            tmp_targets_lst = targets.tolist()
            targets_lst += tmp_targets_lst
            ###

    MAE = metrics.mean_absolute_error(output_lst, targets_lst)
    sMAEP = smape(np.array(output_lst), np.array(targets_lst))
    RMSE = math.sqrt(metrics.mean_squared_error(output_lst, targets_lst))
    R2 = spearmanr(output_lst, targets_lst)

    logging.info("@@@@@@@@@@@ Validation @@@@@@@@@@@@")
    logging.info(f'MAE: {MAE:.3f}')
    logging.info(f'sMAPE: {sMAEP:.3f}%')
    logging.info(f'RMSE: {RMSE:.3f}')
    logging.info(f'R2: ' + str(R2))
    logging.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    return total_mae, output_lst, targets_lst

