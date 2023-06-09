import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_5/basic1.csv")

df.shape

df.head()

df1 = df[df['f4'] == 'ENFJ']
df2 = df[df['f4'] == 'INFP']


abs(df1['f1'].std() - df2['f1'].std())