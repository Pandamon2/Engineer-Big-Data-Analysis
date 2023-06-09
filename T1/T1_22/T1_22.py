import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_22/basic2.csv", parse_dates=['Date'], index_col = 0)

df_w = df.resample('W').sum()
df_w.head()

df_w['Sales'].max()
df_w['Sales'].min()

df_w['Sales'].max() - df_w['Sales'].min()
