import numpy as np
import os
import random
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import logging

from loss import *
from utils import *


def train(args):

    set_seed(args.seed)
    logging.info("Loading Data...")
    terms_scores = []
    with open(args.data_file) as f:
        for line in f.readlines():
            tmp = line.strip('\n').split('\t')
            terms_scores.append(tmp)

    random.shuffle(terms_scores)
    num_terms = len(terms_scores)
    ###

    training_set = terms_scores[:int(num_terms*(args.splits-1)/args.splits)] #  -> training:testing
    testing_set = terms_scores[int(num_terms*(args.splits-1)/args.splits):]
    train_dataset = BertDataset(training_set)
    test_dataset = BertDataset(testing_set)

    all_strings = get_testing_strings(test_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    coffate_fn = collater(tokenizer)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                collate_fn=coffate_fn,
                                drop_last=args.if_drop_last,
                                shuffle=args.if_shuffle)
                    #            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                collate_fn=coffate_fn,
                                drop_last=args.if_drop_last)
                    #            num_workers=args.num_workers)


    logging.info("Init nn...")

    model = BertSST2Model(args.pretrained_model_name)
    model.to(args.device)
    # model = nn.DataParallel(model, device_ids=args.device_ids)
    # model = model.cuda(device=args.device_ids[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*args.num_epochs*args.warmup_portion)\
                                                    , num_training_steps=len(train_loader)*args.num_epochs)
    
    ######
    MAE_loss = nn.L1Loss()
    criterion = BMCLoss(init_noise_sigma=args.init_sigma, device=args.device)
    optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': args.sigma_lr, 'name': 'noise_sigma'})
    #####
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

    logging.info("Training...")
    baseline = args.baseline
    for epoch in range(1, args.num_epochs + 1):
        train_epoch(optimizer, scheduler, criterion, train_loader, model, args.device, epoch, args.check_step)
        test_loss, output_lst, targets_lst \
                                        = test_epoch(MAE_loss, test_loader, model, args.device, epoch)

        ##### Write some samples
        pred_targ = open('Epoch' + str(epoch) + '_' + args.sample_file, "w+")
        pred_targ.write("Epoch: " + str(epoch) + '\n')
        pred_targ.write("STRING|PREDICTIONS|TARGETS\n")
        for l in range(len(output_lst)):
            pred_targ.write(all_strings[l] + '|' + '%.4f' % output_lst[l]\
                                                 + '|' + '%.4f' % targets_lst[l] + '\n')
        pred_targ.close()
        #####
        if test_loss < baseline:
            baseline = test_loss

            # checkpoints_dirname = "Coder_" + timestamp
            checkpoints_dirname = "Biobert_" + timestamp
            os.makedirs(checkpoints_dirname, exist_ok=True)
            save_pretrained(model, optimizer, 
                            checkpoints_dirname + '/checkpoints-{}/'.format(epoch))


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def main():
    logging.basicConfig(
    filename='./logs/bert4nmls.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=3,
                    help="random seed for initialization")

    parser.add_argument(
        "--pretrained_model_name",
        # default="GanjinZero/coder_eng",
        default="dmis-lab/biobert-v1.1",
        type=str,
        help="Bert pretrained model",
    )
    parser.add_argument(
        "--data_file",
        default='./data.csv',
        type=str,
        help="Path to the file of scored strings ",
    )
    parser.add_argument(
        "--sample_file",
        default='sample.txt',
        type=str,
        help="output for storing the prediction results on the testing set for every epoch",
    )

    parser.add_argument("--check_step", default=300, type=int,
                    help="the interval for printing the training loss, *batch")
    parser.add_argument("--num_epochs", default=40, type=int,
                    help="number of epochs for model training")
    parser.add_argument("--splits", default=10, type=int,
                    help="num_training:num_testing = splits-1")
    parser.add_argument("--batch_size", default=256, type=int,
                    help="bacth size")    
    parser.add_argument("--if_shuffle", default=True, type=bool,
                    help="parameter for the dataloader")    
    parser.add_argument("--if_drop_last", default=True, type=bool,
                    help="parameter for the dataloader")  
    parser.add_argument("--device", default="cuda:0", type=str,
                    help="device assigned for modelling")    

    parser.add_argument("--init_sigma", default=1.5, type=float,
                    help="init parameters for BMC loss")    
    parser.add_argument("--sigma_lr", default=0.001, type=float,
                    help="learning rate for the parameters of BMC loss")  

    parser.add_argument("--lr", default=2e-5, type=float,
                    help="learning_rate")  
    parser.add_argument("--warmup_portion", default=0.05, type=float,
                    help="warmup portion for the 'get_linear_schedule_with_warmup'")  
    parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="weigth decay value for the optimizer")  
    parser.add_argument("--baseline", default=1e7, type=float,
                    help="model checkpoint saving initial paramters")  

    args = parser.parse_args()    
    train(args)

if __name__=='__main__':
    main()