import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_18/basic2.csv")
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek
df.head()

df['weekend'] = df['dayofweek'].apply(lambda x: x>=5)
df.head()

weekend_cond = (df['year']==2022) & (df['month'] == 5) & (df['weekend'])
weekday_cond = (df['year']==2022) & (df['month'] == 5) & (~df['weekend'])

weekend = df[weekend_cond]['Sales'].mean()
weekday = df[weekday_cond]['Sales'].mean()

print(round(weekend - weekday, 2))