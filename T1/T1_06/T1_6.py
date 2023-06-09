import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_6/basic1.csv")

# f1 결측치 제거
df = df[~df['f1'].isnull()]
df.head()

df2 = df.groupby(['city','f2']).sum()
df2

df2['f1']

# 조건에 맞는 f1값
df2.iloc[0]['f1']