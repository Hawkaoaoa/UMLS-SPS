from typing_extensions import dataclass_transform
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from scipy.stats import spearmanr
from sklearn import metrics
import math


def plot_length_distribution(data_file):
    len_lst = []
    with open(data_file) as f:

        for i in f.readlines():
            tmp = i.strip('\n').split('\t')[0]
            len_lst.append(len(tmp))

    sns.kdeplot(data=len_lst, shade=True)
    plt.title("Length Distribution")
    plt.savefig("length_distrib.png", dpi=300)      


def plot_score_distribution(data_file):
    value_lst = []
    num_data = 0
    with open(data_file) as f:

        for i in f.readlines():
            num_data += 1
            tmp = i.strip('\n').split('\t')[1]
            if float(tmp) > 1:
                print(float(tmp))
            value_lst.append(float(tmp))

    sns.kdeplot(data=value_lst, shade=True)
    plt.title("Values Distribution")
    plt.savefig("values_distrib.png",dpi=300)  
    print(num_data)


def analyze_pred_targ_mapping(table_file, sample_file):
    table_file = open(table_file, "r+")
    query_table = []
    for line in table_file:
        line = line.strip('\n')
        tmp_lst = line.split(',')

        tmp = []
        the_SAV = tmp_lst[0]
        tmp.append(the_SAV)
        the_TTY = tmp_lst[1]
        tmp.append(the_TTY)
        the_SUPP = tmp_lst[2]
        if the_SUPP == 'Yes':
            tmp.append('Y')
        else:
            tmp.append('N')
        
        query_table.append(tmp)

    scores_file = open(sample_file, "r+")
    predictions = []
    targets = []
    cur_line = 0
    for i in scores_file.readlines():
        if cur_line <= 1:
            cur_line += 1
            continue
        line = i.strip('\n').split('|')
        predictions.append(float(line[1]))
        targets.append(float(line[2]))


    ## Plots
    sns.kdeplot(data=predictions, label="pred", shade=True)
    sns.kdeplot(data=targets, label="targ", shade=True)
    plt.title("Mapping")
    plt.legend()
    # sns.jointplot(x=predictions, y=targets, kind='kde')
    plt.savefig("results.png",dpi=300)

    # Results statistics
    MAE = metrics.mean_absolute_error(predictions, targets)
    # MAPE = metrics.mean_absolute_percentage_error(predictions, targets)
    def smape(A, F):
        return 100/len(A) * sum(2 * abs(F - A) / (abs(A) + abs(F)))
    sMAEP = smape(np.array(predictions), np.array(targets))
    RMSE = math.sqrt(metrics.mean_squared_error(predictions, targets))
    R2 = spearmanr(predictions, targets)

    print(f'MAE: {MAE:.3f}')
    print(f'sMAPE: {sMAEP:.3f}%')
    print(f'RMSE: {RMSE:.3f}')
    print(f'R2: ', R2)



if __name__=='__main__':

    plot_length_distribution(data_file="../data.csv")

    # plot_score_distribution(data_file="../data.csv")

    # analyze_pred_targ_mapping(table_file="../2022AA.csv", sample_file="../Epoch1_sample.txt")