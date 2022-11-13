import pandas as pd
import sys

filename = sys.argv[1]

df = pd.read_csv(filename, header=None)

s = ''
for el in df[1].value_counts()[[i for i in range(5)]]:
    s += str(el) + '\n'
s = s[:-1]

with open("level_1_output.txt", 'w') as f:
    f.write(s)
