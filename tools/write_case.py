import random
import numpy as np
import torch


low_score_file = open("./low_score_strings.txt", "w+")
with open("./data.csv", "r+") as f:
    for line in f.readlines():
        line = line.strip('\n')
        
        if float(line.split('\t')[1]) < 0.5:
            low_score_file.write(line.split('\t')[0] + '\t' + line.split('\t')[1] + '\n')

low_score_file.close()