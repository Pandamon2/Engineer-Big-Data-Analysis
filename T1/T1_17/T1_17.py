import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_17/basic2.csv")

df.head()
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.info()

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

df.head()

cond = (df['year'] == 2022) & (df['month'] == 5)
print(df[cond]['Sales'].median())