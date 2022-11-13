import pandas as pd
import sys

input_filename = sys.argv[1]
outputh_path = sys.argv[2]

df = pd.read_csv(input_filename, header=None)
df = df.rename(columns={0: 'x', 1: 'y'})

df['len'] = df['x'].apply(lambda x: len(x))
df = df[df['len'] == 30]

s = ''
for el in df['y'].value_counts()[[i for i in range(5)]]:
    s += str(el) + '\n'
s = s[:-1]

with open("level_2_output.txt", 'w') as f:
    f.write(s)
df.to_csv(f'{outputh_path}/cleaned_train.csv')
