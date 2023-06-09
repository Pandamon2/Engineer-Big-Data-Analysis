import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_8/basic1.csv")
df.shape

# 누적합
df2 = df[df['f2'] == 1]['f1'].cumsum()
df2 = df2.fillna(method = 'bfill')
df2

print(df2.mean())