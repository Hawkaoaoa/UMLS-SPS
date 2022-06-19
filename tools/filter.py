import re


def check_en_str(string):

    pattern = re.compile('^[A-Za-z0-9.,:;!?()\[\]\-@&%#\^\*|\"_*"\' ]+$')
    if pattern.fullmatch(string):
        return True
    else:
        return False


new_file = open("./data.csv", "w+")
tmp_dict = {}
with open("./raw_data.csv", "r+") as f:
    for line in f.readlines():
        line = line.strip('\n')
        
        the_string = line.split('\t')[0]
        the_score = float(line.split('\t')[1])

        if not check_en_str(the_string):  # only English strings are kept
            continue
        if the_string in tmp_dict.keys():  #  Same strings but with different CUI should be unified
            if the_score > tmp_dict[the_string]:
                tmp_dict[the_string] = the_score
        else:
            tmp_dict[the_string] = the_score       

for k, v in tmp_dict.items():
    new_file.write(k + '\t' + str(v) + '\n')
new_file.close()
        