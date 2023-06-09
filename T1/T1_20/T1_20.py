import pandas as pd

b1 = pd.read_csv("D:/실습/T1/T1_20/basic1.csv")
b3 = pd.read_csv("D:/실습/T1/T1_20/basic3.csv")

b1.head()
b3.head()

df = pd.merge(left = b1,right = b3, how = "left", on = "f4")
df.head()
df.tail()

df.isnull().sum()

print(df.shape)
df = df.dropna(subset = ['r2'])
print(df.shape)

df = df.reset_index()
print(df.iloc[:20]['f2'].sum())