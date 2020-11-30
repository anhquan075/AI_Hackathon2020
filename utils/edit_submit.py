import os 
import pandas as pd 

class_0_path = "filter_Khoa_ver3.txt"
submission_path = "ver3.txt"

with open(class_0_path) as f:
    content = f.readlines()

content = [x.strip() for x in content]

df = pd.read_csv(submission_path, delimiter="\t", header=None)

for index, row in df.iterrows():
    if row[0] in content:
        with open("new_ver3.txt", "a+") as f:
            f.write('{}\t{}\n'.format(row[0], 0))
    else:
        with open("new_ver3.txt", "a+") as f:
            f.write('{}\t{}\n'.format(row[0], row[1]))