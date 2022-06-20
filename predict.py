import os 
import argparse
import numpy as np
import random

import torch
from transformers import AutoTokenizer
from utils import BertSST2Model


def predict(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    inputs =  tokenizer(args.string,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=100)

    model = BertSST2Model(args.pretrained_model_name).to(args.device)
    params_dict = torch.load(args.pretrained_model_path, map_location=args.device)
    model.load_state_dict(params_dict['Coder'])
    model.eval()

    with torch.no_grad():
        inputs.to(args.device)
        score = model(inputs)
        print('The score for "' + args.string + '" is ' + '%.4f'% float(score) )


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=3,
                    help="random seed for initialization")

    parser.add_argument(
        "--pretrained_model_name",
        default="GanjinZero/coder_eng",
        type=str,
        help="Bert pretrained model",
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="./Coder_06_19_21_08/checkpoints-8/model.pth",
        type=str,
        help="the path to the model fine-tuned for the string scoring",
    )
    parser.add_argument("--device", default="cuda:1", type=str,
                    help="device assigned for modelling") 

    parser.add_argument("--string", default="Diet Drinks", type=str,
                    help="the string which needs to be scored") 

    args = parser.parse_args()   
    predict(args)

if __name__=='__main__':
    main()