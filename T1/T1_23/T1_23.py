import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_23/basic1.csv")
df.tail()

top10 = df['f1'].sort_values(ascending=False).iloc[9]
df['f1'] = df['f1'].fillna(top10)

result1 = df['f1'].median()
result1

print(df.shape)
df = df.drop_duplicates(subset = ['age'])
print(df.shape)

result2 = df['f1'].median()
result2

print(abs(result1 - result2))