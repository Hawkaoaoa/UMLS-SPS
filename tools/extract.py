import pandas as pd


raw_file = pd.read_csv("./MRCONSO.RRF", sep='|', header=None)
targeted_file = raw_file.iloc[:, [0, 11, 12, 16, 14]]

targeted_file.to_csv("./MRCONSO.csv", sep='|')

with open('./MRCONSO.csv', mode='r', encoding='utf-8') as f: # delete 1st row
    line = f.readlines()  #
    try:
        line = line[1:]  
        f = open('./MRCONSO.csv', mode='w', encoding='utf-8')  
        f.writelines(line)    
        f.close()            
    except:
        pass
