import numpy as np
import re

######## 
# Current rules
# 1. rankings in the NMLS standard preference table (linear affinity)
# 2. string length ( if > 50, then linear penalty)
# ...
########
table_file = open("./2022AA.csv", "r+")
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

num_standards = len(query_table)
score_unit = 1.0 / num_standards # between 0~1

nmls_file = open("./MRCONSO.csv", "r+")
cutoff_len = 100
penalty_len = 50

scores_file = open("./raw_data.csv", "w+")
line_id = 0
tmp_CUI_set = {}

for line in nmls_file:
    line_id += 1
    line = line.strip('\n')
    tmp_lst = line.split("|")
    
    the_score = 0

    the_id = tmp_lst[0]
    the_CUI = tmp_lst[1]
    the_SAV = tmp_lst[2]
    the_TTY = tmp_lst[3]
    the_SUPP = tmp_lst[4]
    the_string = tmp_lst[5]

    the_len = len(the_string)  # preference for the brief expressions

    if the_len > cutoff_len:
        continue
    
    tmp_table = []
    tmp_table.append(the_SAV)
    tmp_table.append(the_TTY)
    tmp_table.append(the_SUPP)

    if tmp_table in query_table: # the position in the NMLS standard table
        the_ranking = query_table.index(tmp_table)
        the_score = (num_standards-the_ranking) * score_unit
    else:
        the_score = 0

    if the_len > penalty_len:  # an empirical value
        the_score = the_score * (cutoff_len-the_len) / (cutoff_len-penalty_len)
    ####
    if line_id == 1:
        prev_CUI = the_CUI
        tmp_CUI_set[the_string] = the_score
    else:
        ####
        if the_CUI == prev_CUI:
            if the_string in tmp_CUI_set.keys():
                if the_score > tmp_CUI_set[the_string]: # keep only the highest score for each string
                    tmp_CUI_set[the_string] = the_score
                    
            else:
                tmp_CUI_set[the_string] = the_score
        else:  # new CUI comes (this method is valid because the data is all ordered in this way!!!)
            for k, v in tmp_CUI_set.items():
                scores_file.write(k + '\t' + str(v) + '\n')
            tmp_CUI_set = {}
            tmp_CUI_set[the_string] = the_score
            prev_CUI = the_CUI
    ###

    