import pandas as pd

df = pd.read_csv("D:\실습\T1\T1_12\covid-vaccination-vs-death_ratio.csv")
df.shape
df.head()
df2 = df.groupby('country').max()
df2 = df2.sort_values(by = 'ratio',ascending=False)

df2 = df2[1:]

top = df2['ratio'].head(10).mean()
bottom = df2['ratio'].tail(10).mean()

print(round(top - bottom, 1))