import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_24/basic2.csv")
df

df['previous_PV'] = df['PV'].shift(1)
df.head()

df['previous_PV'] = df['previous_PV'].fillna(method = 'bfill')
df.head()

df[(df['Events'] == 1) & (df['Sales'] <= 1000000)]['previous_PV'].sum()